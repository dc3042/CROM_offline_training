# CROM_offline_training

This repository contains the offline training pipeline for [CROM](https://arxiv.org/abs/2206.02607#:~:text=CROM%3A%20Continuous%20Reduced%2DOrder%20Modeling%20of%20PDEs%20Using%20Implicit%20Neural%20Representations,-Peter%20Yichen%20Chen&text=The%20excessive%20runtime%20of%20high,%2Dorder%20modeling%20(ROM).)

## Prerequisites
We assume a fresh install of Ubuntu 20.04. For example,

```
docker run --gpus all --shm-size 128G -it --rm -v $HOME:/home/ubuntu ubuntu:20.04
```

Install python and pip:
```
apt-get update
apt install python3-pip
```

## Dependencies
From the project directory install through pip:

```
pip install .
```

Alternatively, you may install via PyPI directly

```
pip install run_crom
```


## Usage

### Training

```python
run_crom -mode train -d [data directory] -initial_lr [learning rate constant] -epo [epoch sequence] -lr [learning rate scaling sequence] -batch_size [batch size] -lbl [label length] -scale_mlp [network width scale] -ks [kernel size] -strides [stride size] [-siren_dec] [-dec_omega_0 [decoder siren omega]] [-siren_enc] [-enc_omega_0 [encoder siren omega]] 
```

For example 

```python
run_crom -mode train -d /home/ubuntu/sim_data/libTorchFem_data/extreme_pig/test_tension011_pig_long_l-0.01_p2d -lbl 6 -lr 1 0.1 0.05 0.02 0.01 -epo 3000 3000 3000 3000 3000 -siren_dec -batch_size 4 -scale_mlp 64 -dec_omega_0 30 --gpus 1
```

### Testing

```python
run_crom -mode test -m [path to .ckpt file to test]
```

You may also provide any built-in flags for PytorchLightning's [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags)

### Data 
Simulation data should be stored in a directory with the following structure. 
For example, 
```
├───sim_data_parent_directory (contain multiple simulation sequences; each entry in this directory is a simulation sequence)
    ├───sim_seq_ + suffix
        ├───h5_f_0000000000.h5
        ├───h5_f_0000000001.h5
        ├───...
        
    ├───....
```
See SimulationState under simulation.py for the structure of the h5 file.