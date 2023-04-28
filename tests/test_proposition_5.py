import numpy as np
import argparse
import pdb
import nibabel as nib

from digital_diffeomorphism.jacobian_calculator import calc_Ji_3d, calc_Ji_2d
from digital_diffeomorphism.kernel_creator import KernelCreator
from digital_diffeomorphism.utils import identity_grid_like

def test_proposition_5_ndv():
    # Generate a random 100x100x100x3 matrix with values in the range [-10, 10]
    displacement_field = np.random.uniform(low=-10, high=10, size=(100, 100, 100, 3))
    transformation = displacement_field + identity_grid_like(displacement_field)

    jacobian_determinants = {}
    kernel_creator = KernelCreator(ndim=3)
    # calculate the finite difference based jacobian determinants
    for diff_direction in ['000', '+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---']:
        kernels = kernel_creator.create_kernels(diff_direction)
        jacobian_determinants[diff_direction] = calc_Ji_3d(transformation, kernels)

    jacobian_determinants['Ji_intersection'] = np.ones_like(jacobian_determinants['000'])
    for diff_direction in ['+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---']:
        jacobian_determinants['Ji_intersection'] *= (jacobian_determinants[diff_direction] > 0)

    # if a voxel has 'Ji_intersection' positive, the central difference must be positive
    assert np.sum((jacobian_determinants['Ji_intersection'] > 0) * (jacobian_determinants['000'] <= 0)) == 0

def test_proposition_5_nda():
    # Generate a random 1000x1000x2 matrix with values in the range [-5, 5]
    displacement_field = np.random.uniform(low=-5, high=5, size=(1000, 1000, 2))
    transformation = displacement_field + identity_grid_like(displacement_field)

    jacobian_determinants = {}
    kernel_creator = KernelCreator(ndim=2)
    # calculate the finite difference based jacobian determinants
    for diff_direction in ['00', '++', '+-', '-+', '--']:
        kernels = kernel_creator.create_kernels(diff_direction)
        jacobian_determinants[diff_direction] = calc_Ji_2d(transformation, kernels)

    jacobian_determinants['Ji_intersection'] = np.ones_like(jacobian_determinants['00'])
    for diff_direction in ['++', '+-', '-+', '--']:
        jacobian_determinants['Ji_intersection'] *= (jacobian_determinants[diff_direction] > 0)

    # if a voxel has 'Ji_intersection' positive, the central difference must be positive
    assert np.sum((jacobian_determinants['Ji_intersection'] > 0) * (jacobian_determinants['00'] <= 0)) == 0
