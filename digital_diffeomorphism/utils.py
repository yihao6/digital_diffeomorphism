import pdb
import numpy as np

def identity_grid_like(grid):
    """
    Create an identity grid with the same spatial dimension as the given NumPy grid.

    Args:
        grid (np.ndarray): A 2D or 3D grid whose spatial dimension will be used to create the identity grid.

    Returns:
        identity (np.ndarray): An identity grid with the same spatial dimension as the input grid, plus an additional dimension
        for the number of dimensions of the input grid. For example, if the input grid has shape (n1, n2, n3, 3),
        then the output grid will also have shape (n1, n2, n3, 3).

    """
    shape = grid.shape[:-1]

    vectors = [np.arange(0, dim, 1) for dim in shape]

    grids = np.meshgrid(*vectors)
    grids = [np.swapaxes(x, 0, 1) for x in grids]

    identity = np.stack(grids, axis=-1).astype('float32')

    return identity

def identity_grid(shape, padding=0):
    """
    Generate an identity grid of the specified shape with optional padding.
    
    Args:
        shape (tuple): A tuple representing the dimensions of the output grid.
        padding (int, optional, default: 0): The number of paddingto add
        around the grid. 
    
    Returns:
        (numpy.ndarray): An identity grid with the specified shape and padding.
        The output array will have the same number of dimensions as the input shape, and
        the value at each position will correspond to its own index.
    """
    vectors = [np.arange(0-padding, dim+padding, 1) for dim in shape]

    grids = np.meshgrid(*vectors)
    grids = [np.swapaxes(x, 0, 1) for x in grids]

    identity = np.stack(grids, axis=-1).astype('float32')

    return identity

def spatial_padding(transformation):
    """
    Apply spatial padding to a given transformation by extending its dimensions
    and pad around its edges.
    
    Args:
        transformation (np.ndarray): A NumPy array representing the transformation.
    
    Returns:
        (numpy.ndarray): A padded transformation.
    """
    shape = transformation.shape[:-1]
    disp = transformation - identity_grid(shape=shape)

    padding = [(1, 1)] * (transformation.ndim - 1) + [(0, 0)]
    padded_disp = np.pad(disp, padding, mode='edge')

    padded_identity = identity_grid(shape=shape, padding=1)
    return padded_disp + padded_identity

    
