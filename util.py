import numpy as np
import inspect
import time
import math
from pynvml import *
import torch

def getGMem(i):
    h = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(h)
    total_memory_gb_used = round(info.used/1024**3, 1)
    total_memory_gb_total = round(info.total/1024**3, 1)
    total_memory_gb_free = round((info.total-info.used)/1024**3, 1)
    return total_memory_gb_total, total_memory_gb_used, total_memory_gb_free
    
def findEmptyCudaDevice():
    nvmlInit()
    device_id = 0
    free_memory_max = 0
    for i in range(torch.cuda.device_count()):
        total_memory_gb_total, total_memory_gb_used, total_memory_gb_free = getGMem(i)
        if total_memory_gb_free > free_memory_max:
            device_id = i
            free_memory_max = total_memory_gb_free

    if free_memory_max > 4.0:    
        device = torch.device('cuda:'+str(device_id))
    else:
        exit('not enough cuda memory')
    return device

def findEmptyCudaDeviceList(num_gpu):
    nvmlInit()
    used_ram_list = np.array([])
    for i in range(torch.cuda.device_count()):
        total_memory_gb_total, total_memory_gb_used, total_memory_gb_free = getGMem(i)
        used_ram_list = np.append(used_ram_list, total_memory_gb_used)
    
    low_used = np.argsort(used_ram_list)
    gpu_list = low_used[:num_gpu]
    assert(len(gpu_list)==num_gpu)
    return gpu_list.tolist()


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
