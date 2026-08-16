# WSVR_code

> **Paper:**  
> **Weakly Supervised Vehicle Re-Identification with Multi-View Prototype Learning for Cross-Camera Matching**

## Overview

This repository provides the implementation of the proposed **Weakly Supervised Vehicle Re-Identification (WSVR)** framework.

## Requirements

The implementation is tested with the following environment:

- Python 3.10
- PyTorch 1.12.1
- NVIDIA GPU

Please install the required Python packages according to:

```bash
pip install -r requirements.txt
```

PyTorch 1.12.1 should be installed before installing the remaining dependencies.

## Datasets

The experiments are conducted on two widely used vehicle re-identification datasets:

- VeRi-776
- VehicleID

Please download the datasets from their respective official sources and place them under the data/ directory.

The recommended directory structure is:

```bash
WSVR_code/
├── data/
│   ├── VeRi/
│   │   └── ...
│   └── VehicleID/
│       └── ...
├── experiments/
│   └── veri776.yml
│   └── vehicleID.yml
├── train.slurm
├── test.slurm
├── requirements.txt
└── ...
```
The exact directory structure inside data/ may vary depending on the downloaded dataset version. Please ensure that the paths are correctly specified in the corresponding configuration files.

- VeRi-776

The VeRi-776 dataset provides vehicle images captured by multiple non-overlapping cameras. Camera information and trajectory information are used as weak supervision during training.

Please download the dataset from its official source and place it under the data/ directory.

- VehicleID

VehicleID is a large-scale vehicle re-identification dataset containing vehicle images captured under different viewpoints. Please download the dataset from its official source and place it under the data/ directory.

## Dataset Configuration

Before training, please modify the dataset path in:

```bash
experiments/veri776.yml
experiments/vehicleID.yml
```
Set the dataset root directory according to the location of your downloaded dataset.

For example:
```bash
root: /path/to/your/data
```
Please also check other configuration parameters in the YAML file before starting training.

## Training

After preparing the datasets and installing the required environment, training can be started using:

```bash
bash train.slurm
```

The training script is provided in:

```bash
train.slurm
```
Please modify the settings in train.slurm according to your environment.

## Testing

After training, the trained model can be evaluated using:

```bash
bash test.slurm
```
Before testing, please make sure that the checkpoint path and relevant evaluation settings in test.slurm are correctly specified.

## Reproducibility

The implementation is provided to facilitate the reproduction of the experimental results reported in the paper.

The default configuration follows the experimental settings described in the manuscript, including:

- ResNet-50 backbone
- ImageNet-pretrained initialization
- Input resolution of 256 × 256
- 200 training epochs
- SGD optimizer
- Momentum of 0.9
- Weight decay of 5 × 10^-4
- Initial learning rate of 0.01
- Multi-step learning-rate decay
- Temperature parameter of 0.07
- Exponential moving average for memory updating

Please refer to the paper for the exact implementation settings.
