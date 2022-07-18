from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import time
import warnings

from run_crom.util import *
from run_crom.cromnet import *
from run_crom.simulation import SimulationDataset

class Exporter(object):
    def __init__(self, weight_path):
        self.weight_path = weight_path
    
    def export(self, ):

        net = CROMnet.load_from_checkpoint(self.weight_path)
        device = findEmptyCudaDevice()

        # tracing
        #print('tracing begin')
        
        net_enc = net.encoder.to(device)
        net_dec = net.decoder.to(device)
        
        #data_list = DataList(net.data_format['data_path'], 1.0)
        #dataset = SimulationDataset(net.data_format['data_path'], data_list.data_list)
        #trainloader = DataLoader(dataset, batch_size=1)
        data_batched = net.example_input_array

        encoder_input = data_batched.to(device)

        state = encoder_input[:,:, :net.data_format['o_dim']]
        x0 = encoder_input[:, :, net.data_format['o_dim']:]

        net_enc_jit = net_enc.to_torchscript(method = 'trace', example_inputs = state, check_trace=True, check_tolerance=1e-20)

        xhat = net_enc.forward(state).detach()
        xhat_jit = net_enc_jit.forward(state)

        assert(torch.norm(xhat-xhat_jit)<1e-10)

        #print("encoder trace finished")

        xhat = xhat.expand(xhat.size(0), net.data_format['npoints'], xhat.size(2))
        x = torch.cat((xhat, x0), 2)
        
        batch_size_local = x.size(0)
        x = x.view(x.size(0)*x.size(1), x.size(2))

        x_original = x

        x = x.detach()
        x.requires_grad_(True)
        q = net_dec(x)
        q = q.view(batch_size_local, -1, q.size(1)).detach()
        
        net_dec_jit = net_dec.to_torchscript(method = 'trace', example_inputs = x, check_trace=True, check_tolerance=1e-20)
        
        q_jit = net_dec_jit.forward(x)
        q_jit = q_jit.view(batch_size_local, -1, q_jit.size(1))

        assert(torch.norm(q-q_jit)<1e-10)

        #print("decoder trace finished")

        encoder_input = data_batched.to(device)
        output_regular, _, _ = net.forward(encoder_input)

        assert(torch.norm(output_regular-q_jit)<1e-10)

        #print("full network trace finished")

        enc_jit_path = os.path.splitext(self.weight_path)[0]+"_enc.pt"
        dec_jit_path = os.path.splitext(self.weight_path)[0]+"_dec.pt"

        print('decoder torchscript path: ', enc_jit_path)
        torch.jit.save(net_enc_jit, enc_jit_path)  
        #net_enc_jit.save(enc_jit_path)
        print('decoder torchscript path: ', dec_jit_path)
        torch.jit.save(net_dec_jit, dec_jit_path)  
        #net_dec_jit.save(dec_jit_path)

        # trace grad
        x = x_original
        num_sample = 10
        x = x[0:num_sample, :]
        
        net_dec_func_grad = NetDecFuncGrad(net_dec)
        net_dec_func_grad.to(device)

        grad, y = net_dec_func_grad(x)
        grad = grad.clone() # output above comes from inference mode, so we need to clone it to a regular tensor
        y = y.clone()
        
        grad_gt, y_gt = net_dec.computeJacobianFullAnalytical(x)
        outputs_local, _, decoder_input = net.forward(encoder_input)
        grad_gt_auto = computeJacobian(decoder_input, outputs_local)
        grad_gt_auto = grad_gt_auto.view(grad_gt_auto.size(0)*grad_gt_auto.size(1), grad_gt_auto.size(2), grad_gt_auto.size(3))
        grad_gt_auto = grad_gt_auto[0:num_sample, :, :]

        criterion = nn.MSELoss()
        assert(criterion(grad_gt_auto, grad_gt)<1e-10)
        assert(criterion(grad, grad_gt)<1e-10)
        assert(criterion(y, y_gt)<1e-10)
        
        # grad, y = net_auto_dec_func_grad(x)
        with torch.jit.optimized_execution(True):
            net_dec_func_grad_jit = net_dec_func_grad.to_torchscript(method = 'trace', example_inputs = x, check_trace=True, check_tolerance=1e-20)
            grad_jit, y_jit = net_dec_func_grad_jit(x)
        
        assert(torch.norm(grad-grad_jit)<1e-10)
        assert(torch.norm(y-y_jit)<1e-10)

        #print("decoder gradient trace finished")

        dec_func_grad_jit_path = os.path.splitext(self.weight_path)[0]+"_dec_func_grad.pt"
        print('decoder gradient torchscript path: ', dec_func_grad_jit_path)
        net_dec_func_grad_jit.save(dec_func_grad_jit_path)

        net_dec_func_grad.cpu()
        net_dec_func_grad_jit = net_dec_func_grad.to_torchscript(method = 'trace', example_inputs = x, check_trace=True, check_tolerance=1e-20)
        dec_func_grad_jit_path = os.path.splitext(self.weight_path)[0]+"_dec_func_grad_cpu.pt"
        #print('decoder gradient torchscript path (cpu): ', dec_func_grad_jit_path)
        net_dec_func_grad_jit.save(dec_func_grad_jit_path)


        net_enc_jit_load = torch.jit.load(enc_jit_path)
        net_dec_jit_load = torch.jit.load(dec_jit_path)

        encoder_input = data_batched.to(device)
        state = encoder_input[:,:, :net.data_format['o_dim']]
        
        xhat_jit_load = net_enc_jit_load.forward(state)
        assert(torch.norm(xhat_jit_load-xhat_jit)<1e-10)

        x = x_original
        q_jit_load = net_dec_jit_load.forward(x)
        q_jit_load = q_jit_load.view(batch_size_local, -1, q_jit_load.size(1))
        assert(torch.norm(q_jit_load-q_jit)<1e-10)

        net_enc.cpu()
        state = state.cpu()
        net_enc_jit = net_enc.to_torchscript(method = 'trace', example_inputs = state, check_trace=True, check_tolerance=1e-20)
        enc_jit_path = os.path.splitext(self.weight_path)[0]+"_enc_cpu.pt"
        print('encoder torchscript path (cpu): ', enc_jit_path)
        net_enc_jit.save(enc_jit_path)

        net_dec.cpu()
        x = x.cpu()
        net_dec_jit = net_dec.to_torchscript(method = 'trace', example_inputs = x, check_trace=True, check_tolerance=1e-20)
        dec_jit_path = os.path.splitext(self.weight_path)[0]+"_dec_cpu.pt"
        print('decoder torchscript path (cpu): ', dec_jit_path)
        net_dec_jit.save(dec_jit_path)


class CustomCheckPointCallback(ModelCheckpoint):

    CHECKPOINT_NAME_LAST='{epoch}-{step}'

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

        filename = self.last_model_path

        print("\nmodel path: " + filename)

        ex = Exporter(filename)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex.export()
    

class EpochTimeCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
    def on_train_epoch_end(self, trainer, pl_module):
        self.log("epoch_time", (time.time() - self.start_time), prog_bar=True)

class LitProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
