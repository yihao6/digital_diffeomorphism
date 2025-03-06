"""Core functions for non-diffeomorphic volume and non-diffeomorphic area
Author: Yihao Liu <yihao.liu@vanderbilt.edu>
Created on: March 2025
"""
import argparse
import nibabel as nib
import numpy as np
import torch
import os
import logging

from digital_diffeomorphism.jacobian_calculator_torch import (
                    calc_jacdets_3d,
                    calc_jacdets_2d
)

from digital_diffeomorphism.non_diffeomorphic_measures import (
                    calc_non_diffeomorphic_volume,
                    calc_non_diffeomorphic_voxels,
                    calc_non_diffeomorphic_area,
                    calc_non_diffeomorphic_pixels
)

from digital_diffeomorphism.utils import identity_grid_like, spatial_padding


def ndv(trans, mask=None, is_disp=False, use_gpu=False):
    # convert displacement field to deformation field if necessary
    if is_disp:
        trans += identity_grid_like(trans)

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    jacdets = calc_jacdets_3d(trans, device)

    if mask is not None:
        mask = (mask > 0).astype('float32')
    else:
        mask = np.ones_like(trans[..., 0])

    total_voxels = np.sum(mask)

    non_diff_volume_map = calc_non_diffeomorphic_volume(jacdets, mask)
    non_diff_volume = np.sum(non_diff_volume_map)
    non_diff_volume_pct = non_diff_volume / total_voxels * 100

    non_diff_voxels_map = calc_non_diffeomorphic_voxels(jacdets, mask)
    non_diff_voxels = np.sum(non_diff_voxels_map)
    non_diff_voxels_pct = non_diff_voxels / total_voxels * 100

    return non_diff_volume, non_diff_volume_pct, non_diff_volume_map, non_diff_voxels, non_diff_voxels_pct, non_diff_voxels_map

def nda(trans, mask=None, is_disp=False, use_gpu=False):
    # convert displacement field to deformation field if necessary
    if is_disp:
        trans += identity_grid_like(trans)

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    jacdets = calc_jacdets_2d(trans, device)

    if mask is not None:
        mask = (mask > 0).astype('float32')
    else:
        mask = np.ones_like(trans[..., 0])

    total_pixels = np.sum(mask)

    non_diff_area_map = calc_non_diffeomorphic_area(jacdets, mask)
    non_diff_area = np.sum(non_diff_area_map)
    non_diff_area_pct = non_diff_area / total_pixels * 100

    non_diff_pixels_map = calc_non_diffeomorphic_pixels(jacdets, mask)
    non_diff_pixels = np.sum(non_diff_pixels_map)
    non_diff_pixels_pct = non_diff_pixels / total_pixels * 100

    return non_diff_area, non_diff_area_pct, non_diff_area_map, non_diff_pixels, non_diff_pixels_pct, non_diff_pixels_map
