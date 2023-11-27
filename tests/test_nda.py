import numpy as np
import argparse
import nibabel as nib
import torch

from digital_diffeomorphism.jacobian_calculator_torch import calc_jacdets_2d
from digital_diffeomorphism.utils import identity_grid_like, identity_grid, spatial_padding
from digital_diffeomorphism.non_diffeomorphic_measures import calc_non_diffeomorphic_area

class TestExample():
    def setup_method(self, method):
        transformation = identity_grid(shape=[3, 3])
        transformation[1, 1, :] = [0, 2]

        self.jacdets = calc_jacdets_2d(transformation, torch.device("cpu"))

    def test_central_difference(self):
        target = np.array([[1, 0.5, 1], [1.5, 1, 0.5], [1, 1.5, 1]])
        assert np.array_equal(self.jacdets['00'], target)

    def test_forward_forward_difference(self):
        target = np.array([[1, 0, 1], [2, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacdets['++'], target)

    def test_forward_backward_difference(self):
        target = np.array([[1, 0, 1], [1, 3, 0], [1, 1, 1]])
        assert np.array_equal(self.jacdets['+-'], target)

    def test_backward_forward_difference(self):
        target = np.array([[1, 1, 1], [2, -1, 1], [1, 2, 1]])
        assert np.array_equal(self.jacdets['-+'], target)

    def test_backward_backward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 0], [1, 2, 1]])
        assert np.array_equal(self.jacdets['--'], target)

    def test_non_diffeomorphic_area(self):
        mask = np.ones(shape=[3, 3])
        non_diff_area = calc_non_diffeomorphic_area(self.jacdets, mask)
        target = np.array([[0, 0, 0], [0, 1/4, 0], [0, 0, 0]])
        assert np.array_equal(non_diff_area, target)

class TestIdentity:
    def setup_method(self, method):
        transformation = identity_grid(shape=[3, 3])

        self.jacdets = calc_jacdets_2d(transformation, torch.device("cpu"))

    def test_central_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacdets['00'], target)

    def test_forward_forward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacdets['++'], target)

    def test_forward_backward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacdets['+-'], target)

    def test_backward_forward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacdets['-+'], target)

    def test_backward_backward_difference(self):
        target = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        assert np.array_equal(self.jacdets['--'], target)

    def test_non_diffeomorphic_area(self):
        mask = np.ones(shape=[3, 3])
        non_diff_area = calc_non_diffeomorphic_area(self.jacdets, mask)
        target = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert np.array_equal(non_diff_area, target)
