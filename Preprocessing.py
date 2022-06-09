import os
import numpy as np
import torch

class Preprocessing(object):
    def __init__(self, sim_dataset):
        self.sim_dataset = sim_dataset

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
            # read
            with open(preprocessed_file, 'rb') as f:
                self.mean_q = np.load(f)
                self.std_q = np.load(f)
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

    def computeMeanAndStdX(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_x.npy')
        if os.path.exists(preprocessed_file):
            # read
            with open(preprocessed_file, 'rb') as f:
                self.mean_x = np.load(f)
                self.std_x = np.load(f)
        else:
            xs = map(self.gX, range(len(self.sim_dataset)))
            xs = np.vstack(xs)
            self.mean_x = np.mean(xs, axis=0)
            self.std_x = np.std(xs, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean_x)
                np.save(f, self.std_x)
    
    def computeMinAndMaxX(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax_x.npy')
        if os.path.exists(preprocessed_file):
            # read
            with open(preprocessed_file, 'rb') as f:
                self.min_x = np.load(f)
                self.max_x = np.load(f)
        else:
            xs = map(self.gX, range(len(self.sim_dataset)))
            xs = np.vstack(xs)
            self.min_x = np.min(xs, axis=0)
            self.max_x = np.max(xs, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.min_x)
                np.save(f, self.max_x)

    def computeStandardizeTransformation(self):
        
        self.computeMeanAndStdQ()
        self.computeMeanAndStdX()
        self.computeMinAndMaxX()
        return {'mean_q': self.mean_q, 'std_q': self.std_q, 'mean_x': self.mean_x, 'std_x': self.std_x, 'min_x': self.min_x, 'max_x': self.max_x}
