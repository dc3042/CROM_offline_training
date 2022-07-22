from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import time
import warnings

from util import *
from CROMnet import *
from SimulationDataset import *
from Exporter import *


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