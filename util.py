import numpy as np
import inspect
import time
import math
from Callbacks import *
from FindEmptyCuda import *
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy

def getTime():
    return time.strftime("%Y%m%d-%H%M%S")


def generateEPOCHS(learning_rates, epochs):
    
    assert(len(learning_rates)==len(epochs))
    accumulated_epochs = [0]
    accumulated_epochs.extend(np.cumsum(epochs))
    EPOCH_SIZE = accumulated_epochs[-1]

    return learning_rates, accumulated_epochs

def get_validArgs(cls, args):

    params = vars(args)
    valid_kwargs = inspect.signature(cls.__init__).parameters
    network_kwargs = {name: params[name] for name in valid_kwargs if name in params}
    
    return network_kwargs

def conv1dLayer(l_in, ks, strides):
    return math.floor(float(l_in -(ks-1)-1)/strides + 1)

def prepare_Trainer(args):

    output_path = './outputs'
    time_string = getTime()

    weightdir = output_path + '/weights/' + time_string
    checkpoint_callback = CustomCheckPointCallback(verbose=True, dirpath=weightdir, filename='{epoch}-{step}')

    logdir = output_path + '/logs'
    logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name='', version=time_string, log_graph=False)

    callbacks=[checkpoint_callback]

    trainer = Trainer.from_argparse_args(args, gpus=findEmptyCudaDeviceList(args.gpus), default_root_dir=output_path, callbacks=callbacks, logger=logger, max_epochs= np.sum(args.epo), log_every_n_steps=1, strategy=DDPStrategy(find_unused_parameters=False))

    return trainer
