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

# Instantiate the container once and perform the setup with GPU support
srun --ntasks=1 --mpi=pmix --container-image=${CTRIMG} --container-name=nemo_container --container-mounts=/lustre/fsw/general_infra-rd_gsw/theo/nemo-cb/srivatsan:/workspace \
        bash -c 'cp /workspace/callbacks/*.py /opt/NeMo/nemo/lightning/pytorch/callbacks/ && cd /opt/NeMo && pip install .'

# Reuse the instantiated container to run the final command using all tasks with GPU support
srun --mpi=pmix --container-name=nemo_container --container-mounts=/lustre/fsw/general_infra-rd_gsw/theo/nemo-cb/srivatsan:/workspace \
        python /workspace/cloudai_nemorun.py --yes --factory llama3_8b log.ckpt.save_last=False data.global_batch_size=32 trainer.val_check_interval=1000 trainer.callbacks="combined_callbacks" trainer.max_steps=100 data.seq_length=4096 trainer.strategy.tensor_model_parallel_size=2 trainer.strategy.context_parallel_size=1
