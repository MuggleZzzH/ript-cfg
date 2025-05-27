from pathlib import Path
import os
import sys
import torch
from peft import LoraConfig, get_peft_model
from typing import Optional, Union, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
from transformers.cache_utils import Cache

def sdpa_forward_patched(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """This is a patched version of the forward method for LlamaSdpaAttention. 
    It is used to handle the case where the input_embeddings has longer sequence length Used in efficient OpenVLA-OFT forwarding."""
    if output_attentions:
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # In case static cache is used, it is an instance attribute.
    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # In case we are not compiling, we may set `causal_mask` to None, which is required to dispatch to SDPA's Flash Attention 2 backend, rather
    # relying on the `is_causal` argument.

    # NOTE (Moo Jin):
    # We must set `is_causal` == False (to disable the default lower triangular causal mask) and `attn_mask` to the correct attention mask.
    #
    # This is a bit tricky given that we have pad tokens due to collation with samples of different lengths. If there weren't any pad tokens,
    # we could just set `attn_mask` to all zeros, which represents a non-causal bi-directional mask (i.e. attend to all tokens). However,
    # given that we have pad tokens, we need to actually mask out the pad tokens (i.e. set certain attention weights to negative infinity)
    # so that we don't attend to pad tokens.
    #
    # The default causal mask logic produces a `causal_mask` where the lower triangle is all zeros if there are no pad tokens in the sample,
    # and the rest of the elements are negative infinity.
    # If there are K pad tokens in the sample, then the last K columns in `causal_mask` (which has shape (B, 1, seq_len, seq_len)) are
    # additionally negative infinity.
    # Example:
    #
    # No pad tokens:
    #   0 -inf -inf
    #   0   0  -inf
    #   0   0    0
    # 1 pad token:
    #   0 -inf -inf
    #   0   0  -inf
    #   0   0  -inf
    # 2 pad tokens:
    #   0 -inf -inf
    #   0 -inf -inf
    #   0 -inf -inf
    #
    # Intuitively, this stops the last few tokens from attending to the last K tokens which are pad tokens.
    #
    # Okay so the above is what the default causal mask logic returns. But what we want is a mask that is all 0s except for the positions corresponding to pad tokens.
    # What we want is illustrated below.
    #
    # No pad tokens:
    #   0   0    0
    #   0   0    0
    #   0   0    0
    # 1 pad token:
    #   0   0  -inf
    #   0   0  -inf
    #   0   0  -inf
    # 2 pad tokens:
    #   0 -inf -inf
    #   0 -inf -inf
    #   0 -inf -inf
    #
    # Trick: To get this mask, we just take the last row of the old `causal_mask` and duplicate it all the way through to get the new mask. You can see that this trick
    # works by looking at the matrices above.
    # NOTE (Shuhan): we added this to handle the case where the input_embeddings has longer sequence length.
    if causal_mask is not None:
        D = causal_mask.shape[-1]
        last_row = causal_mask[:, :, -1, :].clone()
        new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
        causal_mask = new_mask
        # print('causal_mask.shape', causal_mask.shape)
        q_len = query_states.shape[2]
        # print('q_len', q_len)
        causal_mask = causal_mask[:, :, -q_len:, :]
        # print('causal_mask.shape after', causal_mask.shape)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=False,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

print('Patching LlamaSdpaAttention.forward with custom forward')
LlamaSdpaAttention.forward = sdpa_forward_patched

from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_model,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from prismatic.models.action_heads import LaplaceScaleHead


from dataclasses import dataclass
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    pretrained_head_checkpoint: Union[str, Path] = "" # Pretrained head checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = 'libero_spatial'  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    seed: int = 7                                    # Random Seed (for reproducibility)

    device: str = "cuda"
    
    log_scale_clip: Optional[Tuple[float, float]] = None


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    
    print(model.norm_stats.keys())
    print(unnorm_key)

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)

    # Get Scale Header
    scale_head = LaplaceScaleHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7)

    if cfg.pretrained_head_checkpoint:
        print("Loading pretrained scale header checkpoint from ", cfg.pretrained_head_checkpoint)
        weights = torch.load(cfg.pretrained_head_checkpoint, map_location='cpu')
        scale_head.trunk.load_state_dict(weights['scale_trunk'])
        scale_head.log_b_head.load_state_dict(weights['log_b_head'])

    scale_head = scale_head.to(torch.bfloat16).to(model.device)

    return model, action_head, scale_head, proprio_projector, noisy_action_projector, processor


class OpenVLA_OFT_Policy:
  def __init__(self, pretrained_checkpoint, header_checkpoint, task_suite_name, lora_rank, lora_dropout, lora_adaptor_ckpt, device_id=0, seed=7, fix_scale_head=False, log_scale_clip=None):
    cfg = GenerateConfig()
    cfg.pretrained_checkpoint = pretrained_checkpoint
    cfg.pretrained_head_checkpoint = header_checkpoint
    cfg.task_suite_name = task_suite_name.lower()
    cfg.lora_rank = lora_rank
    cfg.lora_dropout = lora_dropout
    cfg.unnorm_key = cfg.task_suite_name
    cfg.device = 'cuda:' + str(device_id)

    print(f"Rank {device_id} setting seed to {seed + device_id}")
    set_seed_everywhere(seed + device_id)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model, action_head, scale_head, proprio_projector, _, processor = initialize_model(cfg)
    self.processor = processor
    self.action_head = action_head
    self.scale_head = scale_head
    self.proprio_projector = proprio_projector
    self.cfg = cfg
    self.log_scale_clip = log_scale_clip
    cfg.log_scale_clip = log_scale_clip

    if lora_adaptor_ckpt is not None:
        adaptor_path = lora_adaptor_ckpt
        self.model = get_peft_model(model, lora_config)
        self.model.load_adapter(adaptor_path, adapter_name="default")
        print(f"Loaded LORA adaptor from {adaptor_path}")

        header_path = os.path.join(adaptor_path, 'openvla_headers.pt')
        header_weights = torch.load(header_path, map_location='cpu')
        self.action_head.load_state_dict(header_weights['action_header'])
        self.scale_head.load_state_dict(header_weights['scale_header'])
        print('Loaded action header and scale header weights from ', header_path)
        del header_weights
    else:
        self.model = get_peft_model(model, lora_config)

    self.model.print_trainable_parameters()
    self.device = self.model.device
    self.model.eval()

    trainable_params = {}
    trainable_params['model'] = [param for param in self.model.parameters() if param.requires_grad]
    
    trainable_params['header'] = [param for param in self.action_head.parameters() if param.requires_grad]

    if not fix_scale_head:
      print("Training scale head")
      trainable_params['header'] += [param for param in self.scale_head.parameters() if param.requires_grad]
    else:
      print("Fixing scale head")

    self.trainable_params = trainable_params

    if torch.distributed.is_initialized():
       self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[device_id])
       self.action_head = torch.nn.parallel.DistributedDataParallel(self.action_head, device_ids=[device_id])
       self.scale_head = torch.nn.parallel.DistributedDataParallel(self.scale_head, device_ids=[device_id])
       self.proprio_projector = torch.nn.parallel.DistributedDataParallel(self.proprio_projector, device_ids=[device_id])
