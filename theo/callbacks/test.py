from typing import Optional

from lightning.pytorch.callbacks.callback import Callback

from nemo.utils import logging


class TestCallback(Callback):
    def __init__(self):
        logging.info(f'TestCallback created')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> Optional[int]:
        logging.info(f'TestCallback on_train_batch_start')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        logging.info(f'TestCallback on_train_batch_end')
