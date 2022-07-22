from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import _METRIC

import time
import warnings
from typing import Dict
from copy import deepcopy

from util import *
from CROMnet import *
from SimulationDataset import *
from Exporter import *


class CustomCheckPointCallback(ModelCheckpoint):

    CHECKPOINT_NAME_LAST='{epoch}-{step}'

    # Override to get step value from trainer instead of logger_connector
    def _monitor_candidates(self, trainer: "pl.Trainer") -> Dict[str, _METRIC]:
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = (
            epoch.int() if isinstance(epoch, torch.Tensor) else torch.tensor(trainer.current_epoch)
        )

        monitor_candidates["step"] = torch.tensor(trainer.global_step - 1)
        return monitor_candidates

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