import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import numpy as np
import re
import os 
import random
import math
import h5py
from util import *

'''
Simulation Dataset
'''

class SimulationDataset(Dataset):
    def __init__(self, data_path, data_list):
        self.data_list = data_list
        self.data_path = data_path

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        filename = self.data_list[idx]
        sim_data = SimulationState(filename)

        x = sim_data.x[:]
        q = sim_data.q[:]
        time = sim_data.t
        
        x = torch.from_numpy(x).float()
        q = torch.from_numpy(q).float()

        encoder_input = torch.cat((q, x), 1)

        data_item = {'filename': sim_data.filename, 'x': x,
                     'q': q,
                     'encoder_input': encoder_input, 'time': time}
    
        if hasattr(sim_data, 'faces'):
            faces = torch.from_numpy(sim_data.faces).float()
            data_item['faces'] = faces

        return data_item

class SimulationState(object):
    def __init__(self, filename, readfile=True, input_x=None, input_q=None, input_t=None, label=None):
        self.filename = filename
        if readfile:
            with h5py.File(self.filename, 'r') as h5_file:
                self.x = h5_file['/x'][:]
                self.x = np.array(self.x.T)
                self.q = h5_file['/q'][:]
                self.q = np.array(self.q.T)
                self.t = h5_file['/time'][0][0]
                if '/f_tensor' in h5_file:
                    f_tensor_col_major = h5_file['/f_tensor'][:]
                    f_tensor_col_major = np.array(f_tensor_col_major.T)
                    self.f_tensor = f_tensor_col_major.reshape(
                        -1, 3, 3).transpose(0, 2, 1)
                if '/faces' in h5_file:
                    self.faces = h5_file['/faces'][:]
                    self.faces = self.faces.T
        else:
            if input_x is None:
                print('must provide a x if not reading from file')
                exit()
            if input_q is None:
                print('must provide a q if not reading from file')
                exit()
            if input_t is None:
                print('must provide a t if not reading from file')
                exit()
            self.x = input_x
            self.q = input_q
            self.t = input_t
            self.label = label
    
    def write_to_file(self, filename=None):
        if filename:
            self.filename = filename
        print('writng sim state: ', self.filename)
        dirname = os.path.dirname(self.filename)
        os.umask(0)
        os.makedirs(dirname, 0o777, exist_ok=True)
        with h5py.File(self.filename, 'w') as h5_file:
            dset = h5_file.create_dataset("x", data=self.x.T)
            dset = h5_file.create_dataset("q", data=self.q.T)
            self.t = self.t.astype(np.float64)
            dset = h5_file.create_dataset("time", data=self.t)
            if self.label is not None:
                label = self.label.reshape(-1, 1)
                label = label.astype(np.float64)
                dset = h5_file.create_dataset("label", data=label)
        
        if hasattr(self, 'faces'):
            filename_obj = os.path.splitext(self.filename)[0]+'.obj'
            print('writng sim state obj: ', filename_obj)
            obj_loader = ObjLoader()
            obj_loader.vertices = self.q
            obj_loader.faces = self.faces
            obj_loader.export(filename_obj)


'''
Helper Functions
'''

hprefix = 'h5_f'
config_file_pattern = r'h5_f_(\d+)\.h5'
config_file_matcher = re.compile(config_file_pattern)
dir_pattern = r'sim_seq_(.*?)'
dir_matcher = re.compile(dir_pattern)


class DataList(object):
    def __init__(self, root_dir, train_ratio):
        self.data_list, self.data_train_list, self.data_test_list, self.data_train_dir, self.data_test_dir = obtainFilesRecursively(
            root_dir, train_ratio)

def obtainFilesRecursively(path, train_ratio):
    data_list = []
    data_train_list = []
    data_test_list = []
    data_train_dir = []
    data_test_dir = []

    dir_list = os.listdir(path)

    num_sims = 0
    dir_list_sim = []
    for dirname in dir_list:
        if os.path.isdir(os.path.join(path,dirname)):
            dir_match = dir_matcher.match(
                dirname)
            if dir_match != None:
                num_sims += 1
                dir_list_sim.append(dirname)
    random.seed(0)
    random.shuffle(dir_list_sim)

    train_size = math.ceil(train_ratio * num_sims)
    test_size = num_sims - train_size

    counter = 0
    for dirname in dir_list_sim:
        data_list_local = data_train_list if counter < train_size else data_test_list
        data_dir_local = data_train_dir if counter < train_size else data_test_dir
        data_dir_local.append(os.path.join(path, dirname))
        counter += 1
        for filename in os.listdir(os.path.join(path, dirname)):
            config_file_match = config_file_matcher.match(
                filename)
            if config_file_match is None:
                continue
            # skip files begin
            file_number = int(config_file_match[1])
            # skip files finish
            # print(file_number)
            fullfilename = os.path.join(path, dirname, filename)
            data_list.append(fullfilename)
            data_list_local.append(fullfilename)
        # exit()
    return data_list, data_train_list, data_test_list, data_train_dir, data_test_dir

def convertInputFilenameIntoOutputFilename(filename_in, path_basename):
    basename = os.path.basename(filename_in)

    dirname = os.path.dirname(filename_in)
    pardirname = os.path.dirname(dirname)
    pardirname += '_pred'
    pardirname += '_' + path_basename
    dirname = os.path.basename(dirname)

    return os.path.join(pardirname, dirname, basename)

