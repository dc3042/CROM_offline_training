from SimulationDataset import *
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from typing import Optional


class SimulationDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "path/to/dir", batch_size: int = 32, num_workers=1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_list = DataList(self.data_path, 1.0)
        assert (len(self.data_list.data_list) > 0)

        self.sim_dataset = SimulationDataset(self.data_path, self.data_list.data_list)

    def prepare_data(self,):

        self.computeStandardizeTransformation()

    def setup(self, stage: Optional[str] = None):

        self.store_dataParams() 

    def train_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

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

    def computeStandardizeTransformation(self):
        
        self.computeMeanAndStdQ()
        self.computeMeanAndStdX()
        self.computeMinAndMaxX()
    
    def store_dataParams(self):

        minandmax_x_preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax_x.npy')
        with open(minandmax_x_preprocessed_file, 'rb') as f:
            self.min_x = np.load(f)
            self.max_x = np.load(f)
        
        meanandstd_x_preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_x.npy')
        with open(meanandstd_x_preprocessed_file, 'rb') as f:
            self.mean_x = np.load(f)
            self.std_x = np.load(f)
        
        meanandstd_q_preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_q.npy')
        with open(meanandstd_q_preprocessed_file, 'rb') as f:
            self.mean_q = np.load(f)
            self.std_q = np.load(f)
        
        rank_zero_info('x mean: ' + str(self.mean_x))
        rank_zero_info('x std: ' + str(self.std_x))
        rank_zero_info('x min: ' + str(self.min_x))
        rank_zero_info('x max: ' + str(self.max_x))
        rank_zero_info('q mean: ' + str(self.mean_q))
        rank_zero_info('q std: ' + str(self.std_q))

    def get_dataParams(self,): 
        return {'mean_q': self.mean_q, 'std_q': self.std_q, 'mean_x': self.mean_x, 'std_x': self.std_x, 'min_x': self.min_x, 'max_x': self.max_x}

    def get_dataFormat(self, ):

        example_input_array = torch.unsqueeze(self.sim_dataset[0]['encoder_input'], 0)
        [_, i_dim] = self.sim_dataset[0]['x'].shape
        [npoints, o_dim] = self.sim_dataset[0]['q'].shape

        data_format = {'i_dim': i_dim, 'o_dim': o_dim, 'npoints': npoints, 'data_path': self.data_path}

        return data_format, example_input_array

