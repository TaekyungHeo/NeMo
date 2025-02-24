# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import os
from lightning.pytorch.callbacks import Callback
from nemo.utils import logging
from nemo.utils.get_rank import get_rank
from .chakra_profiler import ChakraProfiler
import torch


class ChakraCallback(Callback):
    """
    A PyTorch Lightning callback for profiling with ChakraProfiler.

    This callback enables profiling for specific steps during training and ensures
    proper initialization and cleanup to avoid memory leaks or duplicate profiling instances.

    Args:
        start_step (int): Global batch step to start profiling.
        end_step (int): Global batch step to end profiling.
        warmup_steps (int): Number of warmup steps before profiling starts.
        active_steps (int): Number of active profiling steps.
        trace_dir (str): Directory where traces will be saved.
        library_path (str): Path to the ChakraProfiler shared library.
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        warmup_steps: int = 0,
        active_steps: int = 1,
        trace_dir: str = None,
        library_path: str = None,
    ):
        logging.info("Initializing ChakraCallback...")

        if not isinstance(start_step, int) or not isinstance(end_step, int):
            raise TypeError(f"start_step and end_step must be integers. Got {type(start_step)}, {type(end_step)}")
        if end_step < start_step:
            raise ValueError("end_step must be greater than or equal to start_step.")
        if not trace_dir or not os.path.isdir(trace_dir):
            raise ValueError(f"Chakra trace output path ({trace_dir}) is not set or does not exist.")
        if not library_path or not os.path.isfile(library_path):
            raise ValueError(f"ChakraProfiler library path ({library_path}) is not set or does not exist.")

        self.start_step = start_step
        self.end_step = end_step
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps

        self.trace_dir = Path(trace_dir)
        self.library_path = Path(library_path)

        self.rank = None
        self.profiler = None
        self.is_profiling = False

        logging.info(
            f"Chakra profiling configured:\n"
            f" - Start Step: {self.start_step}\n"
            f" - End Step: {self.end_step}\n"
            f" - Warmup Steps: {self.warmup_steps}\n"
            f" - Active Steps: {self.active_steps}\n"
            f" - Trace Directory: {self.trace_dir}\n"
            f" - Library Path: {self.library_path}\n"
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        logging.info(f"on_train_batch_start called at global_step {trainer.global_step}")

        if trainer.global_step == self.start_step:
            if self.profiler is None:
                self.rank = get_rank()
                logging.info(f"Instantiating ChakraProfiler at rank {self.rank}...")

                self.profiler = ChakraProfiler(
                    library_path=self.library_path,
                    output_trace_dir=self.trace_dir,
                    num_ranks=torch.distributed.get_world_size(),
                    rank=self.rank,
                    wait_steps=0,
                    warmup_steps=self.warmup_steps,
                    active_steps=self.active_steps,
                )

                logging.info(f"ChakraProfiler instantiated successfully at rank {self.rank}. Entering profiler context...")
                self.profiler.__enter__()

            if self.is_profiling:
                logging.warning(f"Attempted to start Chakra profiler multiple times at step {trainer.global_step}. Skipping.")
                return

            logging.info(f"====== Starting Chakra profiling at global_step {trainer.global_step} ======")
            self.profiler.start_recording()
            self.is_profiling = True

            logging.info(f"Chakra Profiler started successfully at global_step {trainer.global_step}.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        logging.info(f"on_train_batch_end called at global_step {trainer.global_step}")

        if self.is_profiling:
            if trainer.global_step < self.end_step:
                logging.info(f"Executing profiler step at global_step {trainer.global_step}")
                self.profiler.profiler_step()
            else:
                logging.info(f"====== Ending Chakra profiling at global_step {trainer.global_step} ======")
                self._stop_profiler()

    def _stop_profiler(self):
        if self.is_profiling and self.profiler:
            logging.info("Stopping Chakra Profiler...")
            self.profiler.stop_recording()

            logging.info("Exiting profiler context...")
            self.profiler.__exit__(None, None, None)

            self.is_profiling = False
            logging.info("Chakra Profiler stopped successfully.")
