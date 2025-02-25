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
import torch
from lightning.pytorch.callbacks import Callback
from nemo.utils import logging
from nemo.utils.get_rank import get_rank


class ChakraCallback(Callback):
    """
    A PyTorch Lightning callback for profiling with PyTorch’s built-in Profiler and ExecutionTraceObserver.

    This callback enables profiling for specific steps during training using PyTorch’s profiler.
    It also captures detailed execution traces using `ExecutionTraceObserver`.
    It ensures proper cleanup, preventing memory leaks or duplicate profiling instances.

    Args:
        start_step (int): Global batch step to start profiling.
        end_step (int): Global batch step to end profiling.
        warmup_steps (int): Number of warmup steps before profiling starts.
        active_steps (int): Number of active profiling steps.
        trace_dir (str): Directory where traces will be saved.
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        warmup_steps: int = 0,
        active_steps: int = 1,
        trace_dir: str = None,
    ):
        if not isinstance(start_step, int) or not isinstance(end_step, int):
            raise TypeError(f"start_step and end_step must be integers. Got {type(start_step)}, {type(end_step)}")
        if end_step < start_step:
            raise ValueError("end_step must be greater than or equal to start_step.")

        if not trace_dir or not os.path.isdir(trace_dir):
            raise ValueError(f"Chakra trace output path ({trace_dir}) is not set or does not exist.")

        self.start_step = start_step
        self.end_step = end_step
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps

        self.trace_dir = Path(trace_dir)
        self.chakra_host_trace_path = self.trace_dir / f"chakra_host_traces"
        self.chakra_device_trace_path = self.trace_dir / f"chakra_device_traces"

        self.chakra_host_trace_path.mkdir(parents=True, exist_ok=True)
        self.chakra_device_trace_path.mkdir(parents=True, exist_ok=True)

        self.trace_observer = torch.profiler.ExecutionTraceObserver()

        def trace_handler(prof):
            rank = get_rank()
            trace_file = self.chakra_device_trace_path / f"rank-{rank}.json"
            prof.export_chrome_trace(str(trace_file))
            logging.info(f"Kineto trace saved: {trace_file}")

        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=self.warmup_steps, active=self.active_steps),
            on_trace_ready=trace_handler,
            execution_trace_observer=self.trace_observer,
        )

        self.is_profiling = False

        logging.info(
            f"Chakra profiling initialized:\n"
            f" - Start Step: {self.start_step}\n"
            f" - End Step: {self.end_step}\n"
            f" - Warmup Steps: {self.warmup_steps}\n"
            f" - Active Steps: {self.active_steps}\n"
            f" - Trace Directory: {self.trace_dir}"
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        if trainer.global_step == self.start_step:
            if self.is_profiling:
                logging.warning(f"Attempted to start Chakra profiler multiple times at step {trainer.global_step}. Skipping.")
                return

            logging.info(f"====== Start Chakra profiling at global_step {trainer.global_step} ======")

            trace_file = self.chakra_host_trace_path / f"rank-{get_rank()}.json"
            self.trace_observer.register_callback(str(trace_file))

            self.profiler.start()
            self.is_profiling = True

            logging.info(f"Chakra Profiler Started.\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if self.is_profiling:
            if trainer.global_step < self.end_step:
                self.profiler.step()
                logging.info(f"Profiler step executed at global_step {trainer.global_step}")
            else:
                logging.info(f"====== End Chakra profiling at global_step {trainer.global_step} ======")
                self._stop_profiler()

    def _stop_profiler(self):
        if self.is_profiling:
            logging.info("Stopping Chakra Profiler...")
            self.profiler.stop()
            self.is_profiling = False

            try:
                logging.info("Unregistering ExecutionTraceObserver...")
                self.trace_observer.unregister_callback()
            except RuntimeError as e:
                logging.warning(f"ExecutionTraceObserver cleanup failed: {e}")
