"""
Example for evaluating the transformations using the digital diffeomorphism critera
and compute the non-diffeomorphic volume.

The transformation is generated by
    im2grid: "Liu, Yihao, et al. "Coordinate Translator for Learning Deformable
            Medical Image Registration." International Workshop on Multiscale
            Multimodal Medical Imaging. Springer, Cham, 2022."
for Learn2Reg 2021 Challenge Task03.

Download the example transformations at:
    https://iacl.ece.jhu.edu/index.php?title=Digital_diffeomorphism
You need to specify the data folder ('--dataroot') and whether to apply mask ('--mask').

Example:
    python example.py --dataroot ./data/ --mask

------------------------------------------------------------------------------------
If you would like to apply this code on your own transformations, make sure that
the input to the function "calc_jac_dets" are deformation fields not displacement fields.
You can use the function "get_identity_grid" to easily convert between the two.

If you use this code please cite the following paper:
    Liu, Yihao, et al. "On Finite Difference Jacobian Computation in Deformable
    Image Registration." arXiv preprint arXiv:2212.06060 (2022).

Licensed under GNU General Public License v3.0
"""
import nibabel as nib
import numpy as np
import os
import scipy.ndimage
import argparse
from tqdm import tqdm
import pdb
from glob import glob
from digital_diffeomorphism.calc_jac_dets import calc_jac_dets, get_identity_grid, calc_measurements

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True, help="path to data folder")
    parser.add_argument("--mask", action='store_true', help="use masks")
    args = parser.parse_args()

    # transformations path
    trans_paths = sorted(glob(os.path.join(args.dataroot, 'trans', '*')))
    # masks path
    if args.mask:
        mask_paths = sorted(glob(os.path.join(args.dataroot, 'masks/**/aligned_seg4.nii.gz')))
    else:
        mask_paths = [None for i in range(len(trans_paths))]

    non_diff_voxels = non_diff_tetrahedra = non_diff_volume = total_voxels = np.empty(0)
    with tqdm(total=len(trans_paths)) as pbar:
        for trans_path, mask_path in zip(trans_paths, mask_paths):
            # load transformation from file
            trans = np.load(trans_path)['arr_0'].astype('float32')

            # upsample the input by a scale of 2 because the transformation was downsampled
            # before submitting to the learn2reg 2021 task03 challenge
            trans = np.array([scipy.ndimage.zoom(trans[i], 2, order=2) for i in range(3)])

            # convert displacement field to deformation field
            trans += get_identity_grid(trans)

            # calculate the Jacobian determinants
            jac_dets = calc_jac_dets(trans)

            # load the label image for mask
            if mask_path:
                mask = nib.load(mask_path).get_fdata().astype('float32')
                mask = (mask[1:-1,1:-1,1:-1] > 0).astype('float32') # remove boundary voxels
            else:
                mask = np.ones_like(trans[0,1:-1,1:-1,1:-1])
            total_voxels = np.append(total_voxels, np.sum(mask))

            # calculate non-diffeomorphic voxels, non-diffeomorphic tetrahedra,
            # and non-diffeomorphic volume
            measurements = calc_measurements(jac_dets, mask)
            non_diff_voxels = np.append(non_diff_voxels, measurements[0])
            non_diff_tetrahedra = np.append(non_diff_tetrahedra, measurements[1])
            non_diff_volume = np.append(non_diff_volume, measurements[2])

            pbar.update(1)

    print('Average Non-diffeomorphic Voxels: {:.2f}({:.2f})({:.2f}%)'.format(
            np.mean(non_diff_voxels),
            np.std(non_diff_voxels),
            np.mean(non_diff_voxels / total_voxels) * 100
        ))
    print('Average Non-diffeomorphic Tetrahedra: {:.2f}({:.2f})({:.2f}%)'.format(
            np.mean(non_diff_tetrahedra),
            np.std(non_diff_tetrahedra),
            np.mean(non_diff_tetrahedra / total_voxels) * 100
        ))
    print('Average Non-diffeomorphic Volume: {:.2f}({:.2f})({:.2f}%)'.format(
            np.mean(non_diff_volume),
            np.std(non_diff_volume),
            np.mean(non_diff_volume / total_voxels) * 100
        ))
