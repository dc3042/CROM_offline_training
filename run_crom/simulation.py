import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os 
import math
import h5py

from run_crom.util import *

'''
Simulation DataModule
'''

class SimulationDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "path/to/dir", batch_size: int = 32, num_workers=1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_list = DataList(self.data_path, 1.0)
        assert (len(self.data_list.data_list) > 0)

        self.sim_dataset = SimulationDataset(self.data_path, self.data_list.data_list)
        self.computeStandardizeTransformation()
        #self.store_dataParams()

    def train_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def gQ(self, idx):
        data_item = self.sim_dataset[idx]
        q = data_item['q']
        return q

    def gX(self, idx):
        data_item = self.sim_dataset[idx]
        x = data_item['x']
        return x

    def computeMeanAndStdQ(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_q.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            qs = map(self.gQ, range(len(self.sim_dataset)))
            qs = np.vstack(qs)
            self.mean_q = np.mean(qs, axis=0)
            self.std_q = np.std(qs, axis=0)
            # process stage with zero std
            for i in range(len(self.std_q)):
                if self.std_q[i] < 1e-12:
                    self.std_q[i] = 1

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean_q)
                np.save(f, self.std_q)
        
        with open(preprocessed_file, 'rb') as f:
            self.mean_q = np.load(f)
            self.std_q = np.load(f)

    def computeMeanAndStdX(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_x.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            xs = map(self.gX, range(len(self.sim_dataset)))
            xs = np.vstack(xs)
            self.mean_x = np.mean(xs, axis=0)
            self.std_x = np.std(xs, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean_x)
                np.save(f, self.std_x)
        
        with open(preprocessed_file, 'rb') as f:
            self.mean_x = np.load(f)
            self.std_x = np.load(f)
    
    def computeMinAndMaxX(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax_x.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            xs = map(self.gX, range(len(self.sim_dataset)))
            xs = np.vstack(xs)
            self.min_x = np.min(xs, axis=0)
            self.max_x = np.max(xs, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.min_x)
                np.save(f, self.max_x)
        
        with open(preprocessed_file, 'rb') as f:
            self.min_x = np.load(f)
            self.max_x = np.load(f)

    def computeStandardizeTransformation(self):
        
        self.computeMeanAndStdQ()
        self.computeMeanAndStdX()
        self.computeMinAndMaxX()
    
    def get_dataParams(self,): 

        return {'mean_q': self.mean_q, 'std_q': self.std_q, 'mean_x': self.mean_x, 'std_x': self.std_x, 'min_x': self.min_x, 'max_x': self.max_x}

    def get_dataFormat(self, ):

        example_input_array = torch.unsqueeze(self.sim_dataset[0]['encoder_input'], 0)
        [_, i_dim] = self.sim_dataset[0]['x'].shape
        [npoints, o_dim] = self.sim_dataset[0]['q'].shape

        data_format = {'i_dim': i_dim, 'o_dim': o_dim, 'npoints': npoints, 'data_path': self.data_path}

        return data_format, example_input_array

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

'''
Simulation State
'''

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
Obj Loader
'''

# from: https://inareous.github.io/posts/opening-obj-using-py
# also checkout: https://pypi.org/project/PyWavefront/

class ObjLoader(object):
    def __init__(self, fileName=None):
        self.vertices = []
        self.faces = []
        ##
        if fileName:
            try:
                f = open(fileName)
                for line in f:
                    if line[:2] == "v ":
                        index1 = line.find(" ") + 1
                        index2 = line.find(" ", index1 + 1)
                        index3 = line.find(" ", index2 + 1)

                        vertex = (float(line[index1:index2]), float(
                            line[index2:index3]), float(line[index3:-1]))
                        self.vertices.append(vertex)

                    elif line[0] == "f":
                        string = line.replace("//", "/")
                        ##
                        i = string.find(" ") + 1
                        face = []
                        for item in range(string.count(" ")):
                            if string.find(" ", i) == -1:
                                face.append(string[i:-1])
                                break
                            face.append(string[i:string.find(" ", i)])
                            i = string.find(" ", i) + 1
                        ##
                        self.faces.append(tuple(face))

                f.close()
            except IOError:
                print(".obj file not found.")

    def export(self, filename):
        f = open(filename, "w")
        f.write("g ")
        f.write("\n")
        for vertex in self.vertices:
            line = "v " + " " + \
                str(vertex[0]) + " " + \
                str(vertex[1]) + " " + str(vertex[2])
            f.write(line)
            f.write("\n")
        f.write("g ")
        f.write("\n")
        for face in self.faces:
            line = "f " + " " + \
                str(face[0]) + " " + \
                str(face[1]) + " " + str(face[2])
            f.write(line)
            f.write("\n")
        f.close()

