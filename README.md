# U-CDA: Unsupervised Brightness and Color Correction for Geological Core Images

This repository contains the official implementation of the study regarding Unsupervised Brightness and Color Correction for Geological Core Images via Column-wise Distribution Alignment and Intra-image Co-training.

## Overview
High-fidelity optical imagery is a prerequisite for intelligent Digital Core Analysis. The U-CDA framework reformulates the image correction task as an intra-image domain adaptation problem. By leveraging the vertical consistency of geological cores and employing a Multi-Kernel Maximum Mean Discrepancy loss, the feature distributions of the distorted edges are aligned to the distortion-free center. This unsupervised mechanism bypasses the heavy reliance on external paired training data, offering a highly adaptable preprocessing standard for automated geosciences.

## Computational Requirements & Dependencies
* **Operating System:** Linux or Windows
* **Hardware:** A CUDA-enabled GPU is recommended for efficient MMD sampling, while CPU execution remains supported.
* **Environment:** Python 3.10 or later is suggested. Ensure the required packages are installed by running the following command:

```bash
pip install -r requirements.txt
```

## Data Availability
Due to industrial confidentiality agreements associated with the original mining exploration area, the complete raw drilling dataset is restricted from public distribution. To fulfill rigorous reproducibility standards and facilitate community verification, representative mini-datasets are provided:

1. **Synthetic Data (`data/synthetic/`):** Contains degraded core images alongside corresponding ground-truth labels, establishing a solid foundation for quantitative full-reference metric evaluations.
2. **Real Core Data (`data/real_cores/`):** Contains practical unlabeled scanning images to demonstrate real-world generalization capabilities.

## Usage & Evaluation

The U-CDA framework conducts optimization dynamically for each single image. The repository provides a unified script for testing.

### Reproducing Synthetic Benchmarks

To evaluate the algorithm using the provided synthetic dataset and calculate full-reference metrics (including PSNR, SSIM, and CIEDE2000), execute the following command:

```bash
python train.py --mode synthetic --input_dir ./data/synthetic --gt_dir ./data/synthetic/label --output_dir ./results/synthetic_out
```

### Evaluating Real Core Images

To process the representative real core scanning images and calculate unsupervised physical-prior metrics (Std_Y, dE_Side, NIQE), execute the following command:

```bash
python train.py --mode real --input_dir ./data/real_cores --output_dir ./results/real_out
```
