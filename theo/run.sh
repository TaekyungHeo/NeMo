#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH -t 0:20:00
#SBATCH --exclusive
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt
#SBATCH --account=general_infra-rd_gsw
#SBATCH --ntasks-per-node=8

CTRIMG=/lustre/fsw/general_infra-rd_gsw/theo/nccl-test/nemo_24_12_01.sqsh
WORKDIR=/lustre/fsw/general_infra-rd_gsw/theo/nemo-cb/srivatsan
MODE="trace_and_execute"
INTERCEPTOR_LIB_NAME="libinterceptor_${MODE}.so"
MAKE_TARGET="interceptor_${MODE}"

srun --ntasks=1 --mpi=pmix --container-image=${CTRIMG} --container-name=nemo_container --container-mounts=$WORKDIR:/workspace \
    bash -c '
    cd /workspace/maya/maya/interceptor
    rm -rf build
    mkdir -p build && cd build
    cmake ..
    make '"$MAKE_TARGET"' -j
    cp '"$INTERCEPTOR_LIB_NAME"' /workspace/
'

srun --ntasks=1 --mpi=pmix --container-name=nemo_container --container-mounts=$WORKDIR:/workspace \
    bash -c 'cp /workspace/callbacks/*.py /opt/NeMo/nemo/lightning/pytorch/callbacks/ && cd /opt/NeMo && pip install .'

srun --mpi=pmix --container-name=nemo_container --container-mounts=$WORKDIR:/workspace \
    bash -c '
    ls /workspace/ && echo '"$INTERCEPTOR_LIB_NAME"' && \
    MAYA_LOG_LEVEL=DEBUG MAYA_OUTPUT_ROOT=/workspace LD_PRELOAD=/workspace/'"$INTERCEPTOR_LIB_NAME"' \
    python /workspace/cloudai_nemorun.py --yes --factory llama3_8b \
        log.ckpt.save_last=False \
        data.global_batch_size=32 \
        trainer.val_check_interval=1000 \
        trainer.callbacks="combined_callbacks" \
        trainer.max_steps=100 \
        data.seq_length=4096 \
        trainer.strategy.tensor_model_parallel_size=2 \
        trainer.strategy.context_parallel_size=1
'
