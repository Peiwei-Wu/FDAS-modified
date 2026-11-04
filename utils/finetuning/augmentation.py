"""复用nnU-Net v2默认二维增强；只在训练Dataset中调用。"""

import numpy as np
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform


def training_transforms(patch_size):
    rotation = np.pi if max(patch_size) / min(patch_size) <= 1.5 else np.pi / 12
    return ComposeTransforms([
        SpatialTransform(tuple(patch_size), patch_center_dist_from_border=0, random_crop=False,
                         p_elastic_deform=0, p_rotation=0.2, rotation=(-rotation, rotation),
                         p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                         bg_style_seg_sampling=False, border_mode_seg="constant", padding_value_seg=-1),
        RandomTransform(GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1,
                                              synchronize_channels=True), apply_probability=0.1),
        RandomTransform(GaussianBlurTransform(blur_sigma=(0.5, 1), synchronize_channels=False,
                                             synchronize_axes=False, p_per_channel=0.5, benchmark=False), apply_probability=0.2),
        RandomTransform(MultiplicativeBrightnessTransform(multiplier_range=BGContrast((0.75, 1.25)),
                                                         synchronize_channels=False, p_per_channel=1), apply_probability=0.15),
        RandomTransform(ContrastTransform(contrast_range=BGContrast((0.75, 1.25)), preserve_range=True,
                                          synchronize_channels=False, p_per_channel=1), apply_probability=0.15),
        RandomTransform(SimulateLowResolutionTransform(scale=(0.5, 1), synchronize_channels=False,
                                                       synchronize_axes=True, ignore_axes=None,
                                                       p_per_channel=0.5), apply_probability=0.25),
        RandomTransform(GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=1,
                                       synchronize_channels=False, p_per_channel=1, p_retain_stats=1), apply_probability=0.1),
        RandomTransform(GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=0,
                                       synchronize_channels=False, p_per_channel=1, p_retain_stats=1), apply_probability=0.3),
        MirrorTransform(allowed_axes=(0, 1)),
        RemoveLabelTansform(-1, 0),
    ])
