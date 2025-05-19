import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    """
    Helper function to rotate the last dimension of a tensor by splitting
    it into two halves and swapping them (with a sign flip on one half).
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """
    Computes rotary positional embeddings.
    For a head dimension of `head_dim`, it computes sin and cos tensors
    of shape (seq_len, head_dim) and then duplicates them to cover the full head.
    """
    def __init__(self, head_dim, base: float = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).float()  # (seq_len,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim/2)
        sin = freqs.sin()
        cos = freqs.cos()
        # Duplicate sin and cos to match the head dimension (i.e. from head_dim/2 to head_dim)
        sin = torch.cat([sin, sin], dim=-1)
        cos = torch.cat([cos, cos], dim=-1)
        return sin, cos

class RoPEMultiheadAttention(nn.Module):
    """
    A custom multi-head attention module that applies RoPE to the query and key.
    It follows the standard scaled dot-product attention but modifies the q/k vectors.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Combine the projections for q, k, and v
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Initialize the rotary embedding for each head’s dimension
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        query, key, value: shape (B, seq_len, embed_dim) if batch_first=True
        """
        if not self.batch_first:
            # Convert from (seq_len, B, embed_dim) to (B, seq_len, embed_dim)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, seq_len, _ = query.size()
        # Project and split into q, k, v
        qkv = self.qkv_proj(query)  # shape: (B, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(B, seq_len, 3, self.num_heads, self.head_dim)
        # Rearrange to shape: (3, B, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, seq_len, head_dim)

        # Obtain sin/cos for rotary embedding
        sin, cos = self.rope(seq_len, query.device)  # each: (seq_len, head_dim)
        # Reshape for broadcasting over (B, num_heads)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

        # Apply RoPE to q and k
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            # key_padding_mask shape should broadcast to (B, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, seq_len, head_dim)
        # Reassemble heads
        attn_output = attn_output.transpose(1, 2).reshape(B, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        return attn_output

class RoPETransformerEncoderLayer(nn.Module):
    """
    A Transformer encoder layer that uses the RoPEMultiheadAttention module.
    It closely follows the design of nn.TransformerEncoderLayer.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, batch_first, norm_first):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal: bool = False):
        """
        The forward function now accepts an extra keyword argument 'is_causal' to remain 
        compatible with PyTorch's TransformerEncoder API.
        """
        # Option to use “pre‐norm” vs “post‐norm”
        if self.norm_first:
            src2 = self.self_attn(
                self.norm1(src), self.norm1(src), self.norm1(src),
                attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )
            src = src + self.dropout1(src2)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src
