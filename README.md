# CROM_offline_training

This repository contains the offline training pipeline for [CROM](https://arxiv.org/abs/2206.02607#:~:text=CROM%3A%20Continuous%20Reduced%2DOrder%20Modeling%20of%20PDEs%20Using%20Implicit%20Neural%20Representations,-Peter%20Yichen%20Chen&text=The%20excessive%20runtime%20of%20high,%2Dorder%20modeling%20(ROM).)

## Usage

### Training

```python
python run.py -mode train -d [data directory] -initial_lr [learning rate constant] -epo [epoch sequence] -lr [learning rate scaling sequence] -batch_size [batch size] -lbl [label length] -scale_mlp [network width scale] -ks [kernel size] -strides [stride size] [-siren_dec] [-dec_omega_0] [-siren_enc] [-enc_omega_0] 
```

### Testing

```python
python run.py -mode test -m [path to .ckpt file to test]
```

You may also provide any built-in flags for PytorchLightning's [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
