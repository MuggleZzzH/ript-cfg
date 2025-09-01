import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ript.algos.utils.obs_core import CropRandomizer
import einops


class IdentityAug(nn.Module):
    def __init__(self, shape_meta=None, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class TranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic.
    """

    def __init__(
        self,
        shape_meta,
        translation,
    ):
        super().__init__()

        self.randomizers = {}
        self.shape_meta = shape_meta

        for name, input_shape in shape_meta['observation']['rgb'].items():
            input_shape = tuple(input_shape)

            self.pad_translation = translation // 2
            pad_output_shape = (
                input_shape[0],
                input_shape[1] + translation,
                input_shape[2] + translation,
            )

            crop_randomizer = CropRandomizer(
                input_shape=pad_output_shape,
                crop_height=input_shape[1],
                crop_width=input_shape[2],
            )
            self.randomizers[input_shape] = crop_randomizer

    def forward(self, data):
        step_indices = data['step_indices']
        if self.training:

            for name in self.shape_meta['observation']['rgb']:
                obs_key = ['obs', 'next_obs']
                for key in obs_key:
                    if key not in data:
                        continue
                    obs_data = data[key]
                    if key == 'next_obs':
                        x = obs_data[name][:, step_indices]
                        # print('next_obs-x.shape', x.shape)
                    else:
                        x = obs_data[name]
                        # print('obs-x.shape', x.shape)

                    batch_size, temporal_len, img_c, img_h, img_w = x.shape

                    input_shape = (img_c, img_h, img_w)
                    crop_randomizer = self.randomizers[input_shape]

                    x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
                    out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
                    out = crop_randomizer.forward_in(out)
                    out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)
                    
                    if key == 'next_obs':
                        obs_data[name][:, step_indices] = out
                    else:
                        obs_data[name] = out
        return data


class ImgColorJitterAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, data):
        step_indices = data['step_indices']
        if self.training and np.random.rand() > self.epsilon:
            for name in self.shape_meta['observation']['rgb']:
                for key in ['obs', 'next_obs']:
                    if key not in data:
                        continue
                    if key == 'next_obs':
                        x = data[key][name][:, step_indices]
                    else:
                        x = data[key][name]
                    x = self.color_jitter(x)
                    if key == 'next_obs':
                        data[key][name][:, step_indices] = x
                    else:
                        data[key][name] = x
        return data


class ImgColorJitterGroupAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, x):
        raise NotImplementedError
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out


class BatchWiseImgColorJitterAug(torch.nn.Module):
    """
    Color jittering augmentation to individual batch.
    This is to create variation in training data to combat
    BatchNorm in convolution network.
    """

    def __init__(
        self,
        shape_meta,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.1,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon
        self.shape_meta = shape_meta

    def forward(self, data):
        step_indices = data['step_indices']
        if self.training:
            for name in self.shape_meta['observation']['rgb']:
                for key in ['obs', 'next_obs']:
                    if key not in data:
                        continue
                    
                    if key == 'next_obs':
                        x = data[key][name][:, step_indices]
                    else:
                        x = data[key][name]
                    
                    mask = torch.rand((x.shape[0], *(1,)*(len(x.shape)-1)), device=x.device) > self.epsilon
                    
                    jittered = self.color_jitter(x)

                    out = mask * jittered + torch.logical_not(mask) * x
                    
                    if key == 'next_obs':
                        data[key][name][:, step_indices] = out
                    else:
                        data[key][name] = out
                
        return data


class DataAugGroup(nn.Module):
    """
    Add augmentation to multiple inputs
    """

    def __init__(self, aug_list, shape_meta):
        super().__init__()
        aug_list = [aug(shape_meta) for aug in aug_list]
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, data):
        return self.aug_layer(data)
    