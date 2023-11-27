"""Calculate Jacobian determinants
Author: Yihao Liu <yliu236@jhu.edu>
Created on: November 2023
"""
import numpy as np
import torch
import torch.nn.functional as nnf
from torch.linalg import det

from digital_diffeomorphism.utils import spatial_padding

def calc_jacdets_3d(trans, device):

    if trans.ndim != 4 or trans.shape[3] != 3:
        raise ValueError("Input transformation must have shape (H, W, D, 3).")

    trans = spatial_padding(trans)

    kernels = {}
    kwargs = {'dtype': torch.float32, 'device': device}
    kernels['D0x'] = torch.tensor([-0.5,   0, 0.5], **kwargs).view(1, 1, 3, 1, 1)
    kernels['D+x'] = torch.tensor([   0,  -1,   1], **kwargs).view(1, 1, 3, 1, 1)
    kernels['D-x'] = torch.tensor([  -1,   1,   0], **kwargs).view(1, 1, 3, 1, 1)

    kernels['D0y'] = torch.tensor([-0.5,   0, 0.5], **kwargs).view(1, 1, 1, 3, 1)
    kernels['D+y'] = torch.tensor([   0,  -1,   1], **kwargs).view(1, 1, 1, 3, 1)
    kernels['D-y'] = torch.tensor([  -1,   1,   0], **kwargs).view(1, 1, 1, 3, 1)

    kernels['D0z'] = torch.tensor([-0.5,   0, 0.5], **kwargs).view(1, 1, 1, 1, 3)
    kernels['D+z'] = torch.tensor([   0,  -1,   1], **kwargs).view(1, 1, 1, 1, 3)
    kernels['D-z'] = torch.tensor([  -1,   1,   0], **kwargs).view(1, 1, 1, 1, 3)

    # J1*
    kernels['1*xy'] = torch.tensor(
                            [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
                            **kwargs,
    ).reshape(1, 1, 3, 3, 1)
    kernels['1*xz'] = torch.tensor(
                            [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
                            **kwargs,
    ).reshape(1, 1, 3, 1, 3)
    kernels['1*yz'] = torch.tensor(
                            [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
                            **kwargs,
    ).reshape(1, 1, 1, 3, 3)

    # J2*
    kernels['2*xy'] = torch.tensor(
                            [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
                            **kwargs,
    ).reshape(1, 1, 3, 3, 1)
    kernels['2*xz'] = torch.tensor(
                            [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
                            **kwargs,
    ).reshape(1, 1, 3, 1, 3)
    kernels['2*yz'] = torch.tensor(
                            [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
                            **kwargs,
    ).reshape(1, 1, 1, 3, 3)

    for kernel_name, kernel in kernels.items():
        kernels[kernel_name] = kernel.type(torch.float32)

    # combine kernels with the same sizes
    weights = {
        'x': torch.cat([kernels[key] for key in ['D0x', 'D+x', 'D-x']] * 3, dim=0),
        'y': torch.cat([kernels[key] for key in ['D0y', 'D+y', 'D-y']] * 3, dim=0),
        'z': torch.cat([kernels[key] for key in ['D0z', 'D+z', 'D-z']] * 3, dim=0),

        '*xy': torch.cat([kernels[key] for key in ['1*xy', '2*xy']] * 3, dim=0),
        '*xz': torch.cat([kernels[key] for key in ['1*xz', '2*xz']] * 3, dim=0),
        '*yz': torch.cat([kernels[key] for key in ['1*yz', '2*yz']] * 3, dim=0),
    }

    trans = torch.from_numpy(trans).permute(3, 0, 1, 2)
    trans = (trans[None,...]).type(torch.float32).to(device)

    partials = {
        'x': nnf.conv3d(trans, weights['x'], groups=3)[:, :, :, 1:-1, 1:-1],
        'y': nnf.conv3d(trans, weights['y'], groups=3)[:, :, 1:-1, :, 1:-1],
        'z': nnf.conv3d(trans, weights['z'], groups=3)[:, :, 1:-1, 1:-1, :],

        '*xy': nnf.conv3d(trans, weights['*xy'], groups=3)[:, :, :, :, 1:-1],
        '*xz': nnf.conv3d(trans, weights['*xz'], groups=3)[:, :, :, 1:-1, :],
        '*yz': nnf.conv3d(trans, weights['*yz'], groups=3)[:, :, 1:-1, :, :],
    }

    jacobians = {
        '000': torch.stack((
                            partials['x'][:, ::3, ...],
                            partials['y'][:, ::3, ...],
                            partials['z'][:, ::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        '+++': torch.stack((
                            partials['x'][:, 1::3, ...],
                            partials['y'][:, 1::3, ...],
                            partials['z'][:, 1::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        '++-': torch.stack((
                            partials['x'][:, 1::3, ...],
                            partials['y'][:, 1::3, ...],
                            partials['z'][:, 2::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        '+-+': torch.stack((
                            partials['x'][:, 1::3, ...],
                            partials['y'][:, 2::3, ...],
                            partials['z'][:, 1::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        '+--': torch.stack((
                            partials['x'][:, 1::3, ...],
                            partials['y'][:, 2::3, ...],
                            partials['z'][:, 2::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        '-++': torch.stack((
                            partials['x'][:, 2::3, ...],
                            partials['y'][:, 1::3, ...],
                            partials['z'][:, 1::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        '-+-': torch.stack((
                            partials['x'][:, 2::3, ...],
                            partials['y'][:, 1::3, ...],
                            partials['z'][:, 2::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        '--+': torch.stack((
                            partials['x'][:, 2::3, ...],
                            partials['y'][:, 2::3, ...],
                            partials['z'][:, 1::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        '---': torch.stack((
                            partials['x'][:, 2::3, ...],
                            partials['y'][:, 2::3, ...],
                            partials['z'][:, 2::3, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        'j1*': torch.stack((
                            partials['*xy'][:, 0::2, ...],
                            partials['*xz'][:, 0::2, ...],
                            partials['*yz'][:, 0::2, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
        'j2*': torch.stack((
                            partials['*xy'][:, 1::2, ...],
                            partials['*yz'][:, 1::2, ...],
                            partials['*xz'][:, 1::2, ...],
                ), dim=-1).permute(0, 2, 3, 4, 1, 5),
    }

    jacdets = {key: det(jacobians[key]).cpu().numpy().squeeze() for key in jacobians}

    return jacdets

def calc_jacdets_2d(trans, device):

    if trans.ndim != 3 or trans.shape[2] != 2:
        raise ValueError("Input transformation must have shape (H, W, 2).")

    trans = spatial_padding(trans)

    kernels = {}
    kwargs = {'dtype': torch.float32, 'device': device}
    kernels['D0x'] = torch.tensor([-0.5,   0, 0.5], **kwargs).view(1, 1, 3, 1)
    kernels['D+x'] = torch.tensor([   0,  -1,   1], **kwargs).view(1, 1, 3, 1)
    kernels['D-x'] = torch.tensor([  -1,   1,   0], **kwargs).view(1, 1, 3, 1)

    kernels['D0y'] = torch.tensor([-0.5,   0, 0.5], **kwargs).view(1, 1, 1, 3)
    kernels['D+y'] = torch.tensor([   0,  -1,   1], **kwargs).view(1, 1, 1, 3)
    kernels['D-y'] = torch.tensor([  -1,   1,   0], **kwargs).view(1, 1, 1, 3)

    for kernel_name, kernel in kernels.items():
        kernels[kernel_name] = kernel.type(torch.float32)

    # combine kernels with the same sizes
    weights = {
        'x': torch.cat([kernels[key] for key in ['D0x', 'D+x', 'D-x']] * 2, dim=0),
        'y': torch.cat([kernels[key] for key in ['D0y', 'D+y', 'D-y']] * 2, dim=0),
    }

    trans = torch.from_numpy(trans).permute(2, 0, 1)
    trans = (trans[None,...]).type(torch.float32).to(device)

    partials = {
        'x': nnf.conv2d(trans, weights['x'], groups=2)[:, :, :, 1:-1],
        'y': nnf.conv2d(trans, weights['y'], groups=2)[:, :, 1:-1, :],
    }

    jacobians = {
        '00': torch.stack((
                            partials['x'][:, ::3, ...],
                            partials['y'][:, ::3, ...],
                ), dim=-1).permute(0, 2, 3, 1, 4),
        '++': torch.stack((
                            partials['x'][:, 1::3, ...],
                            partials['y'][:, 1::3, ...],
                ), dim=-1).permute(0, 2, 3, 1, 4),
        '+-': torch.stack((
                            partials['x'][:, 1::3, ...],
                            partials['y'][:, 2::3, ...],
                ), dim=-1).permute(0, 2, 3, 1, 4),
        '-+': torch.stack((
                            partials['x'][:, 2::3, ...],
                            partials['y'][:, 1::3, ...],
                ), dim=-1).permute(0, 2, 3, 1, 4),
        '--': torch.stack((
                            partials['x'][:, 2::3, ...],
                            partials['y'][:, 2::3, ...],
                ), dim=-1).permute(0, 2, 3, 1, 4),
    }

    jacdets = {key: det(jacobians[key]).cpu().numpy().squeeze() for key in jacobians}

    return jacdets

