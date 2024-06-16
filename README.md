# Deep Learning for Hypertension forecasting

This repository provides the means of fast prototyping and experimenting with the models for Hypertension forecasting.

## Environment Installation

To prepare all needed libraries, create the environment from the file using Anaconda.
```
conda env create --file environment.yml
```
This environment consists of pytorch with cuda 11.3. If you want to use CPU-only version or other version of CUDA remove the pytorch libraries from the environment file before installation:
```
  - cudatoolkit=11.3.1=h59b6b97_2
...
  - pytorch=1.10.1=py3.9_cuda11.3_cudnn8_0
  - pytorch-mutex=1.0=cuda
...
  - torchaudio=0.10.1=py39_cu113
  - torchvision=0.11.2=py39_cu113
```
Then proceed to install pytorch version that suits your needs using the [Pytorch Quickstart](https://pytorch.org/get-started/locally/)

## Running the training

For debugging purposes use the ```run_multiple_experiments.py``` with the **debug_mode** flag set to **True** and chosen project and experiments that you wish to test.
