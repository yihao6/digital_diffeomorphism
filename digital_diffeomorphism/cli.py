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

from digital_diffeomorphism.core import ndv, nda

def non_diffeomorphic_volume():
    parser = argparse.ArgumentParser(description='Calculate non-diffeomorphic volume.')
    parser.add_argument('trans_file', type=str, help='Path to the transformation Nifti file')
    parser.add_argument('--disp', action='store_true', help='Flag for displacement field input')
    parser.add_argument('--mask_file', type=str, help='Path to the mask Nifti file')
    parser.add_argument('--save_results', action='store_true', help='Save the output nifti in the transformation folder')
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

    if args.mask_file:
        mask = nib.load(args.mask_file).get_fdata()
    else:
        mask = None

    logging.debug('Calculating NDV...')
    nd_volume, nd_volume_pct, nd_volume_map, nd_voxels, nd_voxels_pct, nd_voxels_map = ndv(trans, mask, args.disp, args.gpu)

    print(f'Non-diffeomorphic Volume: {nd_volume:.2f}({nd_volume_pct:.2f}%)')
    print(f'Non-diffeomorphic Voxels: {nd_voxels:.2f}({nd_voxels_pct:.2f}%)')

    if args.save_results:
        save_filename = args.trans_file.replace('.nii.gz', '_non_diff_volume.nii.gz')
        nib.save(nib.Nifti1Image(nd_volume_map, affine), save_filename)
        logging.info(f'Saved non-diffeomorphic volume map to {save_filename}')

        save_filename = args.trans_file.replace('.nii.gz', '_non_diff_voxels.nii.gz')
        nib.save(nib.Nifti1Image(nd_voxels_map, affine), save_filename)
        logging.info(f'Saved non-diffeomorphic voxel map to {save_filename}')

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

    if args.mask_file:
        mask = nib.load(args.mask_file).get_fdata()
    else:
        mask = None

    logging.debug('Calculating NDA...')
    nd_area, nd_area_pct, nd_area_map, nd_pixels, nd_pixels_pct, nd_pixels_map = nda(trans, mask, args.disp, args.gpu)

    print(f'Non-diffeomorphic Area: {nd_area:.2f}({nd_area_pct:.2f}%)')
    print(f'Non-diffeomorphic Pixels: {nd_pixels:.2f}({nd_pixels_pct:.2f}%)')

    if args.save_results:
        save_filename = args.trans_file.replace('.nii.gz', '_non_diff_area.nii.gz')
        nib.save(nib.Nifti1Image(nd_area_map, affine), save_filename)
        logging.info(f'Saved non-diffeomorphic area map to {save_filename}')

        save_filename = args.trans_file.replace('.nii.gz', '_non_diff_pixels.nii.gz')
        nib.save(nib.Nifti1Image(nd_pixels_map, affine), save_filename)
        logging.info(f'Saved non-diffeomorphic pixels map to {save_filename}')

