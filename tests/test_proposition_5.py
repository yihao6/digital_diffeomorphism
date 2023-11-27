import numpy as np
import argparse
import nibabel as nib
import torch

from digital_diffeomorphism.jacobian_calculator_torch import (
            calc_jacdets_3d,
            calc_jacdets_2d
)
from digital_diffeomorphism.utils import identity_grid_like

def test_proposition_5_ndv():
    # Generate a random 100x100x100x3 matrix with values in the range [-10, 10]
    displacement_field = np.random.uniform(low=-10, high=10, size=(100, 100, 100, 3))
    transformation = displacement_field + identity_grid_like(displacement_field)

    # calculate the finite difference based jacobian determinants
    jacdets = calc_jacdets_3d(transformation, torch.device("cpu"))

    jacdets['intersection'] = np.ones_like(jacdets['000'])
    for direction in ['+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---']:
        jacdets['intersection'] *= (jacdets[direction] > 0)

    # if a voxel has 'intersection' positive, the central difference must be positive
    assert np.sum((jacdets['intersection'] > 0) * (jacdets['000'] <= 0)) == 0

def test_proposition_5_nda():
    # Generate a random 1000x1000x2 matrix with values in the range [-5, 5]
    displacement_field = np.random.uniform(low=-5, high=5, size=(1000, 1000, 2))
    transformation = displacement_field + identity_grid_like(displacement_field)

    # calculate the finite difference based jacobian determinants
    jacdets = calc_jacdets_2d(transformation, torch.device("cpu"))

    jacdets['intersection'] = np.ones_like(jacdets['00'])
    for direction in ['++', '+-', '-+', '--']:
        jacdets['intersection'] *= (jacdets[direction] > 0)

    # if a voxel has 'intersection' positive, the central difference must be positive
    assert np.sum((jacdets['intersection'] > 0) * (jacdets['00'] <= 0)) == 0
