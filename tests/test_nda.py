import numpy as np
import argparse
import pdb
import nibabel as nib

from digital_diffeomorphism.jacobian_calculator import calc_Ji_3d, calc_Ji_2d
from digital_diffeomorphism.kernel_creator import KernelCreator
from digital_diffeomorphism.utils import identity_grid_like, identity_grid, spatial_padding
from digital_diffeomorphism.non_diffeomorphic_measures import calc_non_diffeomorphic_area

class TestExample():
    def example_setup(self):
        transformation = identity_grid(shape=[3, 3])
        transformation[1, 1, :] = [0, 2]
        transformation = spatial_padding(transformation)

        self.jacobian_determinants = {}
        kernel_creator = KernelCreator(ndim=2)
        for diff_direction in ['00', '++', '+-', '-+', '--']:
            kernels = kernel_creator.create_kernels(diff_direction)
            self.jacobian_determinants[diff_direction] = calc_Ji_2d(transformation, kernels)

    def test_central_difference(self):
        target = np.array([[1, 0.5, 1], [1.5, 1, 0.5], [1, 1.5, 1]])
        assert np.array_equal(self.jacobian_determinants['00'], target)

    def test_forward_forward_difference(self):
        target = np.array([[1, 0, 1], [2, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacobian_determinants['++'], target)

    def test_forward_backward_difference(self):
        target = np.array([[1, 0, 1], [1, 3, 0], [1, 1, 1]])
        assert np.array_equal(self.jacobian_determinants['+-'], target)

    def test_backward_forward_difference(self):
        target = np.array([[1, 1, 1], [2, -1, 1], [1, 2, 1]])
        assert np.array_equal(self.jacobian_determinants['-+'], target)

    def test_backward_backward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 0], [1, 2, 1]])
        assert np.array_equal(self.jacobian_determinants['--'], target)

    def test_non_diffeomorphic_area(self):
        mask = np.ones(shape=[3, 3])
        non_diff_area = calc_non_diffeomorphic_area(self.jacobian_determinants, mask)
        target = np.array([[0, 0, 0], [0, 1/4, 0], [0, 0, 0]])
        assert np.array_equal(non_diff_area, target)

class TestIdentity:
    def identity_setup(self):
        transformation = identity_grid(shape=[3, 3])
        transformation = spatial_padding(transformation)

        self.jacobian_determinants = {}
        kernel_creator = KernelCreator(ndim=2)
        for diff_direction in ['00', '++', '+-', '-+', '--']:
            kernels = kernel_creator.create_kernels(diff_direction)
            self.jacobian_determinants[diff_direction] = calc_Ji_2d(transformation, kernels)

    def test_central_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacobian_determinants['00'], target)

    def test_forward_forward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacobian_determinants['++'], target)

    def test_forward_backward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacobian_determinants['+-'], target)

    def test_backward_forward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacobian_determinants['-+'], target)

    def test_backward_backward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacobian_determinants['--'], target)

    def test_non_diffeomorphic_area(self):
        mask = np.ones(shape=[3, 3])
        non_diff_area = calc_non_diffeomorphic_area(self.jacobian_determinants, mask)
        target = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert np.array_equal(non_diff_area, target)
