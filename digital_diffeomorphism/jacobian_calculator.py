"""Calculate Jacobian determinants
Author: Yihao Liu <yliu236@jhu.edu>
Created on: April 2023
"""
import numpy as np
import scipy.ndimage

from digital_diffeomorphism.kernel_creator import KernelCreator
from digital_diffeomorphism.utils import spatial_padding

def calc_Ji_3d(trans, kernels):

    gradx = scipy.ndimage.correlate(trans, kernels['x'], mode='nearest')
    grady = scipy.ndimage.correlate(trans, kernels['y'], mode='nearest')
    gradz = scipy.ndimage.correlate(trans, kernels['z'], mode='nearest')

    jacobian = np.stack([gradx, grady, gradz], axis=4)
    determinant = np.linalg.det(jacobian)

    return determinant[1:-1, 1:-1, 1:-1]

def calc_Ji_2d(trans, kernels):

    gradx = scipy.ndimage.correlate(trans, kernels['x'], mode='nearest')
    grady = scipy.ndimage.correlate(trans, kernels['y'], mode='nearest')

    jacobian = np.stack([gradx, grady], axis=3)
    determinant = np.linalg.det(jacobian)

    return determinant[1:-1, 1:-1]

def calc_j1star(trans):

    kernels = {}
    kernels['x']  = np.array([[1, 0, 0],[0, -1, 0],[0, 0, 0]]).reshape(3, 3, 1, 1)
    kernels['y']  = np.array([[1, 0, 0],[0, -1, 0],[0, 0, 0]]).reshape(3, 1, 3, 1)
    kernels['z']  = np.array([[1, 0, 0],[0, -1, 0],[0, 0, 0]]).reshape(1, 3, 3, 1)

    gradx = scipy.ndimage.correlate(trans, kernels['x'], mode='nearest')
    grady = scipy.ndimage.correlate(trans, kernels['y'], mode='nearest')
    gradz = scipy.ndimage.correlate(trans, kernels['z'], mode='nearest')

    jacobian = np.stack([gradx, grady, gradz], axis=4)
    determinant = np.linalg.det(jacobian)

    return determinant[1:-1, 1:-1, 1:-1]

def calc_j2star(trans):

    kernels = {}
    kernels['x']  = np.array([[0, 0, 0],[0, -1, 0],[0, 0, 1]]).reshape(3, 3, 1, 1)
    kernels['y']  = np.array([[0, 0, 0],[0, -1, 0],[0, 0, 1]]).reshape(1, 3, 3, 1)
    kernels['z']  = np.array([[0, 0, 0],[0, -1, 0],[0, 0, 1]]).reshape(3, 1, 3, 1)

    gradx = scipy.ndimage.correlate(trans, kernels['x'], mode='nearest')
    grady = scipy.ndimage.correlate(trans, kernels['y'], mode='nearest')
    gradz = scipy.ndimage.correlate(trans, kernels['z'], mode='nearest')

    jacobian = np.stack([gradx, grady, gradz], axis=4)
    determinant = np.linalg.det(jacobian)

    return determinant[1:-1, 1:-1, 1:-1]

def calc_jacdets_3d(trans, device):

    if trans.ndim != 4 or trans.shape[3] != 3:
        raise ValueError("Input transformation must have shape (H, W, D, 3).")

    trans = spatial_padding(trans)

    jacobian_determinants = {}
    kernel_creator = KernelCreator(ndim=3)
    # calculate the finite difference based jacobian determinants
    for diff_direction in ['000', '+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---']:
        kernels = kernel_creator.create_kernels(diff_direction)
        jacobian_determinants[diff_direction] = calc_Ji_3d(trans, kernels)

    jacobian_determinants['j1*'] = calc_j1star(trans)
    jacobian_determinants['j2*'] = calc_j2star(trans)

    return jacobian_determinants

def calc_jacdets_2d(trans):

    if trans.ndim != 3 or trans.shape[2] != 2:
        raise ValueError("Input transformation must have shape (H, W, 2).")

    trans = spatial_padding(trans)

    jacobian_determinants = {}
    kernel_creator = KernelCreator(ndim=2)
    # calculate the finite difference based jacobian determinants
    for diff_direction in ['00', '++', '+-', '-+', '--']:
        kernels = kernel_creator.create_kernels(diff_direction)
        jacobian_determinants[diff_direction] = calc_Ji_2d(trans, kernels)

    return jacobian_determinants

