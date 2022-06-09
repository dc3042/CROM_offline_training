import numpy as np
from Preprocessing import *
import inspect
import time
import math

def getTime():
    return time.strftime("%Y%m%d-%H%M%S")


def generateEPOCHS(learning_rates, epochs):
    
    assert(len(learning_rates)==len(epochs))
    accumulated_epochs = [0]
    accumulated_epochs.extend(np.cumsum(epochs))
    EPOCH_SIZE = accumulated_epochs[-1]

    return learning_rates, accumulated_epochs

def get_dataParams(master_dataset):

    [_, i_dim] = master_dataset[0]['x'].shape
    [npoints, o_dim] = master_dataset[0]['q'].shape

    data_format = {'i_dim': i_dim, 'o_dim': o_dim, 'npoints': npoints}

    preprop = Preprocessing(master_dataset)
    preprop_params = preprop.computeStandardizeTransformation()

    return data_format, preprop_params

def get_validArgs(cls, args):

    params = vars(args)
    valid_kwargs = inspect.signature(cls.__init__).parameters
    network_kwargs = {name: params[name] for name in valid_kwargs if name in params}
    
    return network_kwargs

def conv1dLayer(l_in, ks, strides):
    return math.floor(float(l_in -(ks-1)-1)/strides + 1)