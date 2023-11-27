"""CLI for non-diffeomorphic volume and non-diffeomorphic area
Author: Yihao Liu <yliu236@jhu.edu>
Created on: April 2023
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


def non_diffeomorphic_volume():
    parser = argparse.ArgumentParser(description='Calculate non-diffeomorphic volume.')
    parser.add_argument('trans_file', type=str, help='Path to the transformation Nifti file')
    parser.add_argument('--disp', action='store_true', help='Flag for displacement field input')
    parser.add_argument('--mask_file', type=str, help='Path to the mask Nifti file')
    parser.add_argument('--save_results', action='store_true', help='Save the output in the transformation folder')
    parser.add_argument('--gpu', action='store_true', help='Flag for using GPU')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    # Set logging level based on verbose argument
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug('Start data loading and pre-processing...')
    # load transformation data
    trans_obj = nib.load(args.trans_file)
    trans, affine = trans_obj.get_fdata(), trans_obj.affine

    # convert displacement field to deformation field if necessary
    if args.disp:
        trans += identity_grid_like(trans)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logging.debug('Calculating Jacobian determinants...')
    jacdets = calc_jacdets_3d(trans, device)

    if args.mask_file:
        mask = nib.load(args.mask_file).get_fdata()
        mask = (mask > 0).astype('float32')
    else:
        mask = np.ones_like(trans[..., 0])

    total_voxels = np.sum(mask)

    logging.debug('Calculating non-diffeomorphic volume and non-diffeomorphic voxels...')
    # non-diffeomorphic volume
    non_diff_volume = calc_non_diffeomorphic_volume(jacdets, mask)
    if args.save_results:
        save_filename = args.trans_file.replace('.nii.gz', '_non_diff_volume.nii.gz')
        nib.save(nib.Nifti1Image(non_diff_volume, affine), save_filename)
        logging.info(f'Saved non-diffeomorphic volume map to {save_filename}')

    non_diff_volume = np.sum(non_diff_volume)
    non_diff_volume_percentage = non_diff_volume / total_voxels * 100
    print(f'Non-diffeomorphic Volume: {non_diff_volume:.2f}({non_diff_volume_percentage:.2f}%)')

    # non-diffeomorphic voxels
    non_diff_voxels = calc_non_diffeomorphic_voxels(jacdets, mask)
    if args.save_results:
        save_filename = args.trans_file.replace('.nii.gz', '_non_diff_voxels.nii.gz')
        nib.save(nib.Nifti1Image(non_diff_voxels, affine), save_filename)
        logging.info(f'Saved non-diffeomorphic voxel map to {save_filename}')

    non_diff_voxels = np.sum(non_diff_voxels)
    non_diff_voxels_percentage = non_diff_voxels / total_voxels * 100
    print(f'Non-diffeomorphic Voxels: {non_diff_voxels:.2f}({non_diff_voxels_percentage:.2f}%)')

def non_diffeomorphic_area():
    parser = argparse.ArgumentParser(description='Calculate non-diffeomorphic area')
    parser.add_argument('trans_file', type=str, help='Path to the transformation Nifti file')
    parser.add_argument('--mask_file', type=str, help='Path to the mask Nifti file')
    parser.add_argument('--disp', action='store_true', help='Flag for displacement field input')
    parser.add_argument('--save_results', action='store_true', help='Save the output in the transformation folder')
    parser.add_argument('--gpu', action='store_true', help='Flag for using GPU')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    # Set logging level based on verbose argument
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug('Start data loading and pre-processing...')
    # load transformation data
    trans_obj = nib.load(args.trans_file)
    trans, affine = trans_obj.get_fdata(), trans_obj.affine

    # convert displacement field to deformation field if necessary
    if args.disp:
        trans += identity_grid_like(trans)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logging.debug('Calculating Jacobian determinants...')
    jacdets = calc_jacdets_2d(trans, device)

    if args.mask_file:
        mask = nib.load(args.mask_file).get_fdata()
        mask = (mask > 0).astype('float32')
    else:
        mask = np.ones_like(trans[..., 0])

    total_pixels = np.sum(mask)

    logging.debug('Calculating non-diffeomorphic area and non-diffeomorphic pixels...')
    # non-diffeomorphic area
    non_diff_area = calc_non_diffeomorphic_area(jacdets, mask)
    if args.save_results:
        save_filename = args.trans_file.replace('.nii.gz', '_non_diff_area.nii.gz')
        nib.save(nib.Nifti1Image(non_diff_area, affine), save_filename)
        logging.info(f'Saved non-diffeomorphic area map to {save_filename}')

    non_diff_area = np.sum(non_diff_area)
    non_diff_area_percentage = non_diff_area / total_pixels * 100
    print(f'Non-diffeomorphic Area: {non_diff_area:.2f}({non_diff_area_percentage:.2f}%)')

    # non-diffeomorphic pixels
    non_diff_pixels = calc_non_diffeomorphic_pixels(jacdets, mask)
    if args.save_results:
        save_filename = args.trans_file.replace('.nii.gz', '_non_diff_pixels.nii.gz')
        nib.save(nib.Nifti1Image(non_diff_pixels, affine), save_filename)
        logging.info(f'Saved non-diffeomorphic pixels map to {save_filename}')

    non_diff_pixels = np.sum(non_diff_pixels)
    non_diff_pixels_percentage = non_diff_pixels / total_pixels * 100
    print(f'Non-diffeomorphic Pixels: {non_diff_pixels:.2f}({non_diff_pixels_percentage:.2f}%)')
