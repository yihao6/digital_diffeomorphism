"""CLI for non-diffeomorphic volume and non-diffeomorphic area
Author: Yihao Liu <yliu236@jhu.edu>
Created on: April 2023
"""
import argparse
import nibabel as nib
import numpy as np

from digital_diffeomorphism.jacobian_calculator import (
    calc_jacobian_determinants_3d, calc_jacobian_determinants_2d)

from digital_diffeomorphism.non_diffeomorphic_measures import (
    calc_non_diffeomorphic_volume, calc_non_diffeomorphic_voxels,
    calc_non_diffeomorphic_area, calc_non_diffeomorphic_pixels)

from digital_diffeomorphism.utils import identity_grid_like, spatial_padding

def non_diffeomorphic_volume():
    parser = argparse.ArgumentParser(description='Calculate non-diffeomorphic volume.')
    parser.add_argument('transformation_file', type=str, help='Path to the transformation Nifti file')
    parser.add_argument('--disp', action='store_true', help='Flag for displacement field input')
    parser.add_argument('--mask_file', type=str, help='Path to the mask Nifti file')
    args = parser.parse_args()

    # load transformation data
    transformation_obj = nib.load(args.transformation_file)
    transformation = transformation_obj.get_fdata()

    # convert displacement field to deformation field if necessary
    if args.disp:
        transformation += identity_grid_like(transformation)

    jacobian_determinants = calc_jacobian_determinants_3d(transformation)

    if args.mask_file:
        mask = nib.load(args.mask_file).get_fdata()
        mask = (mask > 0).astype('float32')
    else:
        mask = np.ones_like(transformation[..., 0])

    total_voxels = np.sum(mask)

    # non-diffeomorphic volume
    non_diff_volume = calc_non_diffeomorphic_volume(jacobian_determinants, mask)
    non_diff_volume = np.sum(non_diff_volume)
    non_diff_volume_percentage = non_diff_volume / total_voxels * 100
    print(f'Non-diffeomorphic Volume: {non_diff_volume:.2f}({non_diff_volume_percentage:.2f}%)')

    # non-diffeomorphic voxels
    non_diff_voxels = calc_non_diffeomorphic_voxels(jacobian_determinants, mask)
    non_diff_voxels = np.sum(non_diff_voxels)
    non_diff_voxels_percentage = non_diff_voxels / total_voxels * 100
    print(f'Non-diffeomorphic Voxels: {non_diff_voxels:.2f}({non_diff_voxels_percentage:.2f}%)')

def non_diffeomorphic_area():
    parser = argparse.ArgumentParser(description='Calculate non-diffeomorphic area')
    parser.add_argument('transformation_file', type=str, help='Path to the transformation Nifti file')
    parser.add_argument('--mask_file', type=str, help='Path to the mask Nifti file')
    parser.add_argument('--disp', action='store_true', help='Flag for displacement field input')
    args = parser.parse_args()

    # load transformation data
    transformation_obj = nib.load(args.transformation_file)
    transformation = transformation_obj.get_fdata()

    # convert displacement field to deformation field if necessary
    if args.disp:
        transformation += identity_grid_like(transformation)

    jacobian_determinants = calc_jacobian_determinants_2d(transformation)

    if args.mask_file:
        mask = nib.load(args.mask_file).get_fdata()
        mask = (mask > 0).astype('float32')
    else:
        mask = np.ones_like(transformation[..., 0])

    total_pixels = np.sum(mask)

    # non-diffeomorphic area
    non_diff_area = calc_non_diffeomorphic_area(jacobian_determinants, mask)
    non_diff_area = np.sum(non_diff_area)
    non_diff_area_percentage = non_diff_area / total_pixels * 100
    print(f'Non-diffeomorphic Area: {non_diff_area:.2f}({non_diff_area_percentage:.2f}%)')

    # non-diffeomorphic pixels
    non_diff_pixels = calc_non_diffeomorphic_pixels(jacobian_determinants, mask)
    non_diff_pixels = np.sum(non_diff_pixels)
    non_diff_pixels_percentage = non_diff_pixels / total_pixels * 100
    print(f'Non-diffeomorphic Pixels: {non_diff_pixels:.2f}({non_diff_pixels_percentage:.2f}%)')
