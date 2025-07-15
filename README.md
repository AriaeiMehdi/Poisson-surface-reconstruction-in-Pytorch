# Poisson-surface-reconstruction-in-Pytorch
This repository is a naive implementation of the "Poisson Surface Reconstruction paper" using PyTorch.

# Overview

This repository provides two utility functions for volumetric reconstruction of 3D point clouds with PyTorch:

fftfreq: Computes properly scaled FFT frequency bins (angular frequencies).

vol: Constructs a binary occupancy grid (characteristic function) of a point cloud via spectral integration, suitable for isosurface extraction (e.g., marching cubes) or direct volume estimation.

# Features

Spectral Reconstruction: Uses FFT-based Poisson integration to reconstruct a continuous indicator function from point samples and normals.

GPU Acceleration: Leveraging PyTorch’s CUDA backend for efficient large-scale voxel grids.

Easy-to-Use: Simple API for frequency computation and volumetric grid generation.
