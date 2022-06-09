from util import *
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
from pytorch_lightning.utilities import rank_zero_info
from datetime import datetime

class CROMnet(pl.LightningModule):
    def __init__(self, data_format, example_input_array, initial_lr, batch_size, lr, epo, lbl, scale_mlp, ks, strides, siren_enc, siren_dec, enc_omega_0, dec_omega_0, verbose, loaded_from=None):
        super(CROMnet, self).__init__()

        self.save_hyperparameters()

        #data specific parameters
        self.data_format = data_format
        self.verbose = verbose #Print data specific parameters or not

        #learning parameters
        self.lr = initial_lr
        self.batch_size = batch_size
        self.learning_rates, self.accumulated_epochs = generateEPOCHS(lr, epo)
        
        #Updated parameters
        self.loaded_from = loaded_from

        #network structure parameters (things we can change for network)
        self.lbllength = lbl
        self.scale_mlp = scale_mlp
        self.ks = ks
        self.strides = strides
        self.siren_enc = siren_enc
        self.siren_dec = siren_dec
        self.enc_omega_0 = enc_omega_0
        self.dec_omega_0 = dec_omega_0      
        
        #Example input array
        self.example_input_array = example_input_array

        self.criterion = nn.MSELoss()

        self.encoder = NetAutoEnc(data_format, self.lbllength, self.ks, self.strides, self.siren_enc, self.enc_omega_0)
        self.decoder = NetAutoDec(data_format, self.lbllength, self.scale_mlp, self.siren_dec, self.dec_omega_0)
        self.sim_state_list = []

    def setup(self, stage):
        self.preprop_params = self.datamodule.get_dataParams()
        self.decoder.invStandardizeQ.set_params(self.preprop_params)
        self.decoder.prepare.set_params(self.preprop_params)
        self.encoder.standardizeQ.set_params(self.preprop_params)

    def training_step(self, train_batch, batch_idx):
        
        encoder_input = train_batch['encoder_input']
        q = train_batch['q']
        outputs_local = self.forward(encoder_input)
        loss = 1000 * self.criterion(outputs_local, q)

        tensorboard_logs = {'train_loss_step': loss}

        self.log_dict(tensorboard_logs, prog_bar=True)
        
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('train_loss_epoch', avg_loss)
        self.log('step', self.current_epoch)

        return 

    def test_step(self, test_batch, batch_idx):

        encoder_input = test_batch['encoder_input']
        q = test_batch['q']
        outputs_local = self.forward(encoder_input)
        loss = 1000 * ((outputs_local - q)**2).mean()
        self.log('test_loss', loss)

        labels = self.forward_enc(encoder_input).detach().cpu().numpy()
        
        batch_size_local = encoder_input.size(0)

        output_regular = outputs_local.detach().cpu().numpy()
        x_regular = test_batch['x'].detach().cpu().numpy().astype(float)
        filenames = test_batch['filename']
        times = test_batch['time'].detach().cpu().numpy()

        # specific to fem dataset
        if 'faces' in test_batch:
            faces = test_batch['faces'].detach().cpu().numpy()

        for i in range(batch_size_local):
            label = labels[i, :]
            input_q = output_regular[i,:,:]
            input_x = x_regular[i,:,:]
            input_t = np.array([[times[i]]])
            filename = filenames[i]
            filename_out = convertInputFilenameIntoOutputFilename(filename, self.path_basename)
            sim_state = SimulationState(filename_out, False, input_x, input_q, input_t, label=label)
            # specific to fem dataset
            if 'faces' in test_batch:
                if self.write_faces:
                    sim_state.faces = faces[i, :, :]
                sim_state.q = sim_state.x + sim_state.q # assume displacenement training data; generalize later
            self.sim_state_list.append(sim_state)

        return loss

    def forward(self, x):

        x = x

        #encoder input
        state = x[:,:, :self.data_format['o_dim']]

        #x0 for decoder
        x0 = x[:, :, self.data_format['o_dim']:]

        #encoder -> xhat
        xhat = self.encoder.forward(state)

        #decoder input
        xhat = xhat.expand(xhat.size(0), self.data_format['npoints'], xhat.size(2))
        x = torch.cat((xhat, x0), 2)
        
        #store original shape for later & reshape for decoder
        batch_size_local = x.size(0)
        x = x.view(x.size(0)*x.size(1), x.size(2))

        #decoder -> x
        x = self.decoder.forward(x)

        #return to original shape
        x = x.view(batch_size_local, -1, x.size(1))

        return x
        
    def print_hyperparameters(self):

        rank_zero_info('\n\n---Data Info---')
        rank_zero_info('data_path: ' + self.data_format['data_path'])
        rank_zero_info('# of points per file: ' + str(self.data_format['npoints']))
        rank_zero_info('# of i_dim: ' + str(self.data_format['i_dim']))
        rank_zero_info('# of o_dim: ' + str(self.data_format['o_dim']))
        rank_zero_info('x mean: ' + str(self.preprop_params['mean_x']))
        rank_zero_info('x std: ' + str(self.preprop_params['std_x']))
        rank_zero_info('x min: ' + str(self.preprop_params['min_x']))
        rank_zero_info('x max: ' + str(self.preprop_params['max_x']))
        rank_zero_info('q mean: ' + str(self.preprop_params['mean_q']))
        rank_zero_info('q std: ' + str(self.preprop_params['std_q']))

        rank_zero_info('\n---Network Info---')

        rank_zero_info('lbllength: ' + str(self.lbllength))
        rank_zero_info('scale_mlp: ' + str(self.scale_mlp))
        rank_zero_info('siren enc: ' + str(self.siren_enc))
        if self.siren_enc:
            rank_zero_info('\tomega_0: ' + str(self.enc_omega_0))
        rank_zero_info('siren_dec: ' + str(self.siren_dec))
        if self.siren_dec:
            rank_zero_info('\tomega_0: ' + str(self.dec_omega_0))
        #rank_zero_info('lambda_f: ' + str(self.lambda_f))
        rank_zero_info("")

    def adaptiveLRfromRange(self, epoch):
        for idx in range(len(self.accumulated_epochs)-1):
            do = self.accumulated_epochs[idx]
            up = self.accumulated_epochs[idx+1]
            if do <= epoch < up:
                return self.learning_rates[idx]
        if epoch == self.accumulated_epochs[-1]: #last epoch
            return self.learning_rates[-1]
        else:
            exit('invalid epoch for adaptiveLRfromRange')

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=self.adaptiveLRfromRange)
        return [optimizer], [scheduler]
        

    def on_train_start(self,):
        if self.verbose:
            self.print_hyperparameters()
    
    def on_test_start(self,):
        if self.verbose:
            self.print_hyperparameters()
    
    def test_epoch_end(self, outputs):

        for sim_state in self.sim_state_list:
            
            sim_state.write_to_file()

class Activation(pl.LightningModule):
    def __init__(self, siren, omega_0):
        super(Activation, self).__init__()
        self.siren = siren
        self.omega_0 = omega_0

    def __call__(self, input):
        if self.siren:
            return self.actSIN(input)
        else:
            return self.actELU(input)
    
    def actELU(self, input):
        return F.elu(input)

    def actSIN(self, input):
        return torch.sin(self.omega_0 * input)

class invStandardizeQ(pl.LightningModule):
    def __init__(self,):
        super(invStandardizeQ, self).__init__()

    def set_params(self, preprop_params):
        self.register_buffer('mean_q_torch', torch.from_numpy(preprop_params['mean_q']).float())
        self.register_buffer('std_q_torch', torch.from_numpy(preprop_params['std_q']).float())

    def __call__(self, q_standardized):
        return self.mean_q_torch  + q_standardized  * self.std_q_torch
    

class Prepare(pl.LightningModule):
    def __init__(self, lbllength, siren):
        super(Prepare, self).__init__()
        self.siren = siren
        self.lbllength = lbllength

    def set_params(self, preprop_params):
        self.register_buffer('min_x_torch', torch.from_numpy(preprop_params['min_x']).float())
        self.register_buffer('max_x_torch', torch.from_numpy(preprop_params['max_x']).float())
        self.register_buffer('mean_x_torch', torch.from_numpy(preprop_params['mean_x']).float())
        self.register_buffer('std_x_torch', torch.from_numpy(preprop_params['std_x']).float())

    def __call__(self, x):
        xhat = x[:,:self.lbllength]
        
        x0 = x[:, self.lbllength:]
        x0 = self.prep(x0)
        
        x = torch.cat((xhat, x0), 1)

        return x

    def clipX(self, x):
        return 2 * (x - self.min_x_torch) / (self.max_x_torch - self.min_x_torch) - 1

    def standardizeX(self, x):
        return (x - self.mean_x_torch) / self.std_x_torch

    def prep(self, x):
        if self.siren:
            return self.clipX(x)
        else:
            return self.standardizeX(x)


class NetAutoDec(pl.LightningModule):
    def __init__(self, data_format, lbllength, scale_mlp, siren, omega_0):
        super(NetAutoDec, self).__init__()

        self.lbllength = lbllength
        self.scale_mlp = scale_mlp        
        self.siren = siren
        self.omega_0 = omega_0

        self.dec0 = nn.Linear(lbllength + data_format['i_dim'], data_format['i_dim'] * scale_mlp)
        self.dec1 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec2 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec3 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec4 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.enc = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['o_dim'])

        self.act = Activation(self.siren, self.omega_0)
        self.invStandardizeQ = invStandardizeQ()
        self.prepare = Prepare(self.lbllength, self.siren)

        self.layers = []
        self.layers.append(self.prepare)
        self.layers.append(self.dec0)
        self.layers.append(self.act)
        self.layers.append(self.dec1)
        self.layers.append(self.act)
        self.layers.append(self.dec2)
        self.layers.append(self.act)
        self.layers.append(self.dec3)
        self.layers.append(self.act)
        self.layers.append(self.dec4)
        self.layers.append(self.act)
        self.layers.append(self.enc)
        self.layers.append(self.invStandardizeQ)

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            for layer in self.layers:
                if layer.__class__.__name__ == 'Linear':

                    random.seed(datetime.now())
                    seed_number = random.randint(0, 100)
                    random.seed(0)
                    torch.manual_seed(seed_number)
                    
                    if self.siren:
                        layer.weight.uniform_(-np.sqrt(6 / layer.in_features) / self.omega_0, 
                                                np.sqrt(6 / layer.in_features) / self.omega_0)
                    else:
                        nn.init.xavier_uniform_(layer.weight)

            if self.siren:
                self.dec0.weight.uniform_(-1 / self.dec0.in_features, 
                                             1 / self.dec0.in_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


class standardizeQ(pl.LightningModule):
    def __init__(self,):
        super(standardizeQ, self).__init__()
    
    def set_params(self, preprop_params):
        self.register_buffer('mean_q_torch', torch.from_numpy(preprop_params['mean_q']).float())
        self.register_buffer('std_q_torch', torch.from_numpy(preprop_params['std_q']).float())

    def __call__(self, q):
        return (q - self.mean_q_torch) / self.std_q_torch


class NetAutoEnc(pl.LightningModule):
    def __init__(self, data_format, lbllength, ks ,strides, siren, omega_0):
        super(NetAutoEnc, self).__init__()
        
        self.lbllength = lbllength
        self.siren = siren
        self.omega_0 = omega_0

        self.ks = ks
        self.strides = strides

        # goal: [o_dim, npoints] --> 32
        self.layers_conv = nn.ModuleList()
        l_in = data_format['npoints']
        layer_cnt = 0

        while True:
            l_out = conv1dLayer(l_in, self.ks, self.strides)
            if data_format['o_dim']*l_out >= 32:
                l_in = l_out
                layer_cnt += 1
                self.layers_conv.append(nn.Conv1d(data_format['o_dim'], data_format['o_dim'], self.ks, self.strides))
            else:
                break
        self.enc10 = nn.Linear(data_format['o_dim']*l_in, 32)
        self.enc11 = nn.Linear(32, lbllength)

        self.standardizeQ = standardizeQ()
        self.act = Activation(self.siren, self.omega_0)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for m in self.children():
                
                random.seed(datetime.now())
                seed_number = random.randint(0, 100)
                random.seed(0)
                torch.manual_seed(seed_number)
                
                if type(m) == nn.Linear:
                    if self.siren:
                        m.weight.uniform_(-np.sqrt(6 / m.in_features) / self.omega_0, 
                                            np.sqrt(6 / m.in_features) / self.omega_0)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                
                elif type(m) == nn.ModuleList:
                    for c in m:
                        if self.siren:
                            c.weight.uniform_(-np.sqrt(6 / self.ks) / self.omega_0, 
                                            np.sqrt(6 / self.ks) / self.omega_0)
                        else:
                            nn.init.xavier_uniform_(c.weight)
                
                torch.manual_seed(torch.initial_seed())

            if self.siren:
                self.layers_conv[0].weight.uniform_(-1 / self.ks, 
                                             1 / self.ks)

    def forward(self, state):
        state = self.standardizeQ(state)

        state = torch.transpose(state, 1, 2)
        
        for layer in self.layers_conv:
            state = self.act(layer(state))
        
        state = torch.transpose(state, 1, 2)

        state = state.reshape(-1, state.size(1)*state.size(2))
        state = self.act(self.enc10(state))
        xhat = self.act(self.enc11(state))
        xhat = xhat.view(xhat.size(0), 1, xhat.size(1))

        return xhat
