"""Calculate non-diffeomorphic measures
Author: Yihao Liu <yliu236@jhu.edu>
Created on: April 2023
"""
import numpy as np

def calc_non_diffeomorphic_volume(jacobian_determinants, mask):
    """
    Calculates the non-diffeomorphic volume using the given jacobian determinants and mask.

    Parameters:
    ----------
    jacobian_determinants : dict
        A dictionary of jacobian determinants for different directions.
    mask : np.ndarray
        A binary mask indicating the region of interest.

    Returns:
    -------
    non_diff_volume : np.ndarray
        A matrix of floating numbers that indicate the non-diffeomorphic volume.
    """
    non_diff_volume = np.zeros_like(jacobian_determinants['000'])
    for diff_direction in ['+++', '++-', '+-+', '+--', 'j1*', 'j2*', '-++', '-+-', '--+', '---']:
        non_diff_volume += -0.5 * np.minimum(jacobian_determinants[diff_direction], 0) * mask / 6

    return non_diff_volume

def calc_non_diffeomorphic_area(jacobian_determinants, mask):
    """
    Calculates the non-diffeomorphic area using the given jacobian determinants and mask.

    Parameters:
    ----------
    jacobian_determinants : dict
        A dictionary of jacobian determinants for different directions.
    mask : np.ndarray
        A binary mask indicating the region of interest.

    Returns:
    -------
    non_diff_area : np.ndarray
        A matrix of floating numbers that indicate the non-diffeomorphic area.
    """
    non_diff_area = np.zeros_like(jacobian_determinants['00'])
    for diff_direction in ['++', '+-', '-+', '--']:
        non_diff_area += -0.5 * np.minimum(jacobian_determinants[diff_direction], 0) * mask / 2

    return non_diff_area

def calc_non_diffeomorphic_voxels(jacobian_determinants, mask):
    """
    Calculates the number of non-diffeomorphic voxels using the given jacobian determinants and mask.

    Parameters:
    ----------
    jacobian_determinants : dict
        A dictionary of jacobian determinants for different directions.
    mask : np.ndarray
        A binary mask indicating the region of interest.

    Returns:
    -------
    non_diff_voxels : np.ndarray
        A binary mask indicating the non-diffeomorphic voxels.
    """
    non_diff_voxels = (jacobian_determinants['000'] <= 0) * mask
    return non_diff_voxels

def calc_non_diffeomorphic_pixels(jacobian_determinants, mask):
    """
    Calculates the number of non-diffeomorphic pixels using the given jacobian determinants and mask.

    Parameters:
    ----------
    jacobian_determinants : dict
        A dictionary of jacobian determinants for different directions.
    mask : np.ndarray
        A binary mask indicating the region of interest.

    Returns:
    -------
    non_diff_pixels : np.ndarray
        A binary mask indicating the non-diffeomorphic pixels.
    """
    non_diff_pixels = (jacobian_determinants['00'] <= 0) * mask
    return non_diff_pixels
