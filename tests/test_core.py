import numpy as np
import argparse
import nibabel as nib
import torch

from digital_diffeomorphism.core import ndv, nda
from digital_diffeomorphism.utils import identity_grid_like, identity_grid, spatial_padding

class TestCoreNDVIdentity:
    def setup_method(self, method):
        transformation = identity_grid(shape=[3, 3, 3])

        self.out = ndv(transformation, is_disp=False)

    def test_nd_area(self):
        target = 0
        assert np.array_equal(self.out[0], target)

    def test_nd_pixels(self):
        target = 0
        assert np.array_equal(self.out[3], target)

class TestNDVDispIdentity:
    def setup_method(self, method):
        transformation = identity_grid(shape=[3, 3, 3])
        disp = transformation - identity_grid_like(transformation)

        self.out = ndv(disp, is_disp=True)

    def test_nd_area(self):
        target = 0
        assert np.array_equal(self.out[0], target)

    def test_nd_pixels(self):
        target = 0
        assert np.array_equal(self.out[3], target)

class TestCoreNDAIdentity:
    def setup_method(self, method):
        transformation = identity_grid(shape=[3, 3])

        self.out = nda(transformation, is_disp=False)

    def test_nd_area(self):
        target = 0
        assert np.array_equal(self.out[0], target)

    def test_nd_pixels(self):
        target = 0
        assert np.array_equal(self.out[3], target)

class TestNDADispIdentity:
    def setup_method(self, method):
        transformation = identity_grid(shape=[3, 3])
        disp = transformation - identity_grid_like(transformation)

        self.out = nda(disp, is_disp=True)

    def test_nd_area(self):
        target = 0
        assert np.array_equal(self.out[0], target)

    def test_nd_pixels(self):
        target = 0
        assert np.array_equal(self.out[3], target)
