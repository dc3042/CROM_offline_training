from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info

class CustomCheckPointCallback(ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

        rank_zero_info("\nModel Version: " + pl_module.logger.version)
