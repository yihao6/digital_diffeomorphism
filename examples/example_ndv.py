import numpy as np
import argparse
import pdb
import nibabel as nib

from digital_diffeomorphism.jacobian_calculator import calc_jacobian_determinants_3d
from digital_diffeomorphism.utils import identity_grid_like
from digital_diffeomorphism.non_diffeomorphic_measures import (
    calc_non_diffeomorphic_volume, calc_non_diffeomorphic_voxels)

parser = argparse.ArgumentParser(description='Calculate non-diffeomorphic volume')
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

jacobian_determinants = calc_jacobian_determinants_3d(transformation)

if args.mask_file:
    mask = nib.load(args.mask_file).get_fdata()
    mask = (mask > 0).astype('float32')
else:
    mask = np.ones_like(transformation[:, :, :, 0])

total_voxels = np.sum(mask)

# non-diffeomorphic volume
non_diff_volume = calc_non_diffeomorphic_volume(jacobian_determinants, mask)
nib.save(nib.Nifti1Image(non_diff_volume, transformation_obj.affine),
         args.transformation_file.replace('.nii.gz', '_non_diff_volume.nii.gz'))
non_diff_volume = np.sum(non_diff_volume)
non_diff_volume_percentage = non_diff_volume / total_voxels * 100
print(f'Non-diffeomorphic Volume: {non_diff_volume:.2f}({non_diff_volume_percentage:.2f}%)')

# non-diffeomorphic voxels
non_diff_voxels = calc_non_diffeomorphic_voxels(jacobian_determinants, mask)
nib.save(nib.Nifti1Image(non_diff_voxels, transformation_obj.affine),
         args.transformation_file.replace('.nii.gz', '_non_diff_voxels.nii.gz'))
non_diff_voxels = np.sum(non_diff_voxels)
non_diff_voxels_percentage = non_diff_voxels / total_voxels * 100
print(f'Non-diffeomorphic Voxels: {non_diff_voxels:.2f}({non_diff_voxels_percentage:.2f}%)')
