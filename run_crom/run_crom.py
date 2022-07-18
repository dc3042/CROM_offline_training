from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy

import argparse
import os

from run_crom.simulation import SimulationDataModule
from run_crom.cromnet import CROMnet
from run_crom.callbacks import *



def prepare_Trainer(args):

    output_path = os.getcwd() + '/outputs'
    time_string = getTime()

    weightdir = output_path + '/weights/' + time_string
    checkpoint_callback = CustomCheckPointCallback(verbose=True, dirpath=weightdir, save_last=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    epoch_timer = EpochTimeCallback()

    custom_progress_bar = LitProgressBar()

    callbacks=[lr_monitor, checkpoint_callback, epoch_timer, custom_progress_bar]

    logdir = output_path + '/logs'
    logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name='', version=time_string, log_graph=False)

    trainer = Trainer.from_argparse_args(args, gpus=findEmptyCudaDeviceList(args.gpus), default_root_dir=output_path, callbacks=callbacks, logger=logger, max_epochs= np.sum(args.epo), log_every_n_steps=1, strategy=DDPStrategy(find_unused_parameters=False))

    return trainer

def main():

    parser = argparse.ArgumentParser(description='Neural Representation training')

    # Mode for script
    parser.add_argument('-mode', help='train or test',
                    type=str, required=True)
    
    # Network arguments
    parser.add_argument('-lbl', help='label length',
                    type=int, required=False, default=6)  
    parser.add_argument('-scale_mlp', help='scale mlp',
                    type=int, required=False, default=10)
    parser.add_argument('-ks', help='scale mlp',
                    type=int, required=False, default=6)
    parser.add_argument('-strides', help='scale mlp',
                    type=int, required=False, default=4)
    parser.add_argument('-siren_dec', help='use siren - decoder',
                        action='store_true')
    parser.add_argument('-siren_enc', help='use siren - encoder',
                        action='store_true')                  
    parser.add_argument('-dec_omega_0', help='dec_omega_0',
                    type=float, required=False, default=30)
    parser.add_argument('-enc_omega_0', help='enc_omega_0',
                        type=float, required=False, default=0.3)
    
    # Network Training arguments
    parser.add_argument('-m', help='path to weight',
                    type=str, required=False)
    parser.add_argument('-d', help='path to the dataset',
                    type=str, required=False)
    parser.add_argument('-verbose', help='verbose',
                        action='store_false')
    parser.add_argument('-initial_lr', help='initial learning rate',
                        type=float, nargs=1, required=False, default=8e-4)
    parser.add_argument('-lr', help='adaptive learning rates',
                    type=float, nargs='*', required=False)
    parser.add_argument('-epo', help='adaptive epoch sizes',
                        type=int, nargs='*', required=False)
    parser.add_argument('-batch_size', help='batch size',
                    type=int, required=False, default=16)

    # Trainer arguments
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    trainer = prepare_Trainer(args)

    if args.mode == "train":
 
        if args.d:

            data_path = args.d

            dm = SimulationDataModule(data_path, args.batch_size, num_workers=64)
            data_format, example_input_array = dm.get_dataFormat()
            preprop_params = dm.get_dataParams()

            network_kwargs = get_validArgs(CROMnet, args)
            net = CROMnet(data_format, preprop_params, example_input_array, **network_kwargs)
        
        else:
            exit('Enter data path')

        trainer.fit(net, dm)
    
    elif args.mode == "test":

        if args.m:

            weight_path = args.m 

            net = CROMnet.load_from_checkpoint(weight_path, loaded_from=weight_path)

            dm = SimulationDataModule(net.data_format['data_path'], net.batch_size, num_workers=64)

        else:
            exit('Enter weight path')

        trainer.test(net, dm)


if __name__ == "__main__":

    main()