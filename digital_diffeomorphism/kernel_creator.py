"""Create the kernels for finite differences
Author: Yihao Liu <yliu236@jhu.edu>
Created on: April 2023
"""
import numpy as np

class KernelCreator:
    """
    A class for creating kernels for finite difference calculations.

    Parameters:
    -----------
    ndim : int
        The number of dimensions in the deformation field.

    Methods:
    --------
    create_kernels(directions)
        Creates finite difference kernels based on the given directions.
    """
    def __init__(self, ndim):
        self.ndim = ndim

    def create_kernels(self, directions):
        """
        Creates finite difference kernels based on the given directions.

        Parameters:
        -----------
        directions : list or tuple
            A list or tuple of length ndim specifying the finite difference directions for each dimension.

        Returns:
        --------
        reshaped_kernels : dict
            A dictionary containing the finite difference kernels for each dimension, reshaped to the appropriate dimensions.
        """
        if len(directions) != self.ndim:
            raise ValueError(f"The 'directions' argument must be a list or tuple of length {self.ndim} for {self.ndim}D deformation field")

        valid_chars = set(['+', '-', '0'])
        for axis, direction in zip(self._get_axes(), directions):
            if direction not in valid_chars:
                raise ValueError(f"Invalid finite difference '{direction}' along axis '{axis}'")

        kernels = {}
        for axis, direction in zip(self._get_axes(), directions):
            if direction == '+':
                # forward difference kernel
                kernels[axis] = np.array([0, -1, 1])

            elif direction == '0':
                # central difference kernel
                kernels[axis] = np.array([-0.5, 0, 0.5])

            elif direction == '-':
                # backward difference kernel
                kernels[axis] = np.array([-1, 1, 0])

        reshaped_kernels = {}
        for axis, kernel in kernels.items():
            reshaped_kernel = kernel.reshape(*self._get_reshape_dims(axis))
            reshaped_kernels[axis] = reshaped_kernel

        return reshaped_kernels

    def _get_axes(self):
        return ['x', 'y', 'z'][:self.ndim]

    def _get_reshape_dims(self, axis):
        reshape_dims = [1] * self.ndim
        axis_index = self._get_axes().index(axis)
        reshape_dims[axis_index] = 3
        return reshape_dims + [1]
