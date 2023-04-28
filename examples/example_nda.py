import numpy as np
import argparse
import pdb
import nibabel as nib

from digital_diffeomorphism.jacobian_calculator import calc_jacobian_determinants_2d
from digital_diffeomorphism.utils import identity_grid_like
from digital_diffeomorphism.non_diffeomorphic_measures import (
    calc_non_diffeomorphic_area, calc_non_diffeomorphic_pixels)

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
nib.save(nib.Nifti1Image(non_diff_area, transformation_obj.affine),
         args.transformation_file.replace('.nii.gz', '_non_diff_area.nii.gz'))
non_diff_area = np.sum(non_diff_area)
non_diff_area_percentage = non_diff_area / total_pixels * 100
print(f'Non-diffeomorphic Area: {non_diff_area:.2f}({non_diff_area_percentage:.2f}%)')

# non-diffeomorphic pixels
non_diff_pixels = calc_non_diffeomorphic_pixels(jacobian_determinants, mask)
nib.save(nib.Nifti1Image(non_diff_pixels, transformation_obj.affine),
         args.transformation_file.replace('.nii.gz', '_non_diff_pixels.nii.gz'))
non_diff_pixels = np.sum(non_diff_pixels)
non_diff_pixels_percentage = non_diff_pixels / total_pixels * 100
print(f'Non-diffeomorphic Pixels: {non_diff_pixels:.2f}({non_diff_pixels_percentage:.2f}%)')
