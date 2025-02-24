# Copyright (c) 2025, Georgia Institute of Technology. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
#
# Modifications and enhancements made by NVIDIA Corporation & Affiliates.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA modifications are proprietary and licensed under the NVIDIA End User
# License Agreement or another applicable agreement. These modifications may
# not be used, copied, modified, or distributed except as expressly authorized
# under such agreements.

import ctypes
import logging
import os
from pathlib import Path


class ChakraProfiler:
    def __init__(
        self,
        library_path: Path,
        output_trace_dir: Path,
        num_ranks: int,
        rank: int,
        wait_steps: int,
        warmup_steps: int,
        active_steps: int,
    ) -> None:
        self.interceptor_lib = self._get_preloaded_library(library_path)
        self.output_trace_dir = output_trace_dir
        self.num_ranks = num_ranks
        self.rank = rank
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps

    @staticmethod
    def _is_library_loaded(lib_path: str) -> bool:
        try:
            with open("/proc/self/maps", "r") as maps_file:
                loaded_libs = maps_file.read()
                return lib_path in loaded_libs or os.path.basename(lib_path) in loaded_libs
        except FileNotFoundError:
            logging.warning("Could not read /proc/self/maps, assuming library is not loaded.")
            return False

    @staticmethod
    def _get_preloaded_library(path: Path) -> ctypes.CDLL:
        lib_path = str(path)

        ld_preload = os.environ.get("LD_PRELOAD", "")
        if lib_path not in ld_preload:
            logging.warning(f"[Profiler] Library {lib_path} is missing from LD_PRELOAD! {ld_preload}")

        if ChakraProfiler._is_library_loaded(lib_path):
            return ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        else:
            raise RuntimeError(f"LD_PRELOAD is set but {lib_path} is not loaded.")

    def enter_profiling(self) -> None:
        logging.info("Entering profiling session.")
        self.interceptor_lib.enter_profiling(
            str(self.output_trace_dir).encode("utf-8"),
            ctypes.c_int(self.num_ranks),
            ctypes.c_int(self.rank),
            ctypes.c_int(self.wait_steps),
            ctypes.c_int(self.warmup_steps),
            ctypes.c_int(self.active_steps),
        )

    def exit_profiling(self) -> None:
        logging.info("Exiting profiling session.")
        self.interceptor_lib.exit_profiling()

    def start_recording(self) -> None:
        logging.info("Starting recording.")
        self.interceptor_lib.start_recording()

    def stop_recording(self) -> None:
        logging.info("Stopping recording.")
        self.interceptor_lib.stop_recording()

    def profiler_step(self) -> bool:
        self.interceptor_lib.profiler_step.restype = ctypes.c_bool
        return self.interceptor_lib.profiler_step()

    def __enter__(self):
        self.enter_profiling()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_profiling()
