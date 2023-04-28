<img src='docs/_static/imgs/example.png' width="1000px"/>

# Digital diffeomorphism volume and Non-diffeomorphic area
This is an implementation of the **digital diffeomorphism volume** and
**non-diffeomorphic area** computation we introduced in our paper:

<a href="https://arxiv.org/abs/2212.06060">Liu, Yihao, et al. "On Finite Difference Jacobian Computation in Deformable Image Registration." arXiv preprint arXiv:2212.06060 (2022).</a>

## Motivation
The Jacobian determinant $|J|$ of spatial transformations is a widely used metric in
deformable image registration, but the details of its computation are often overlooked.
Contrary to what one might expect, the commonly used central difference base $|J|$
does not reflect if the transformation is diffeomorphic or not. We proposed the
definition of digital diffeomorphism that solves several errors that inherent in
the central difference based $|J|$. We further propose to use non-diffeomorphic
volume to measure the irregularity of 3D transformations.

<p align="center">
  <img src='docs/_static/imgs/checkerboard_problem.png' align="center" width="200px"/>
</p>

An failure case of the central difference based $|J|$. The center pixel
has central difference based $|J|=1$ but it is not diffeomorphic. In fact, the
transformation at the center pixel has no effect on the computation of central
difference based $|J|$, even if it moves outside the field of view.
## Getting Started

### Installation
The easiest way to install the package is through the following command:
```
pip install digital-diffeomorphism
```

To install from the source:

- Clone this repo:
```bash
git clone https://github.com/yihao6/digital_diffeomorphism.git
cd digital_diffeomorphism
```
- Install the dependencies:
```bash
python setup.py install
```

### Usage
To evaluate a 3D sampling grid with dimension $H\times W\times D\times 3$
```bash
ndv grid_3d.nii.gz
```
This will calculate
1. non-diffeomorphic volume; and
2. non-diffeomorphic voxels computed by the central difference.

If the transformation is stored as a displacement field:
```bash
ndv disp_3d.nii.gz --disp
```

To evaluate a 2D sampling grid with dimension $H\times W\times 2$
```bash
nda grid_2d.nii.gz
```
This will calculate
1. non-diffeomorphic area; and
2. non-diffeomorphic pixels computed by the central difference.

If the transformation is stored as a displacement field:
```bash
ndv disp_2d.nii.gz --disp
```
### Potential Pitfalls
1. Several packages implement spatial transformations using a normalized sampling grid. For example, <a href="https://arxiv.org/abs/2212.06060">torch.nn.functional.grid_sample</a>. In this package, we use un-normalized coordinates to represent transformations. Therefore, the input sampling grid or displacement field should be in voxel or pixel units. In case the input is normalized, it must be unnormalized prior to using this package.

### Citation
If you use this code, please cite our paper.
```
@article{liu2022finite,
  title={On Finite Difference Jacobian Computation in Deformable Image Registration},
  author={Liu, Yihao and Chen, Junyu and Wei, Shuwen and Carass, Aaron and Prince, Jerry},
  journal={arXiv preprint arXiv:2212.06060},
  year={2022}
}
```
