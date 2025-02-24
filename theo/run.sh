#!/bin/bash

#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

CTRIMG=/mswg2/E2E/theo/nemo-cb/nemo_24_12_01.sqsh

srun --ntasks=1 --mpi=pmix --container-image=${CTRIMG} --container-name=nemo_container --container-mounts=/mswg2/E2E/theo/nemo-cb:/workspace --container-writable --gpus=1 \
bash -c 'cp /workspace/callbacks/*.py /opt/NeMo/nemo/lightning/pytorch/callbacks/ && cd /opt/NeMo && pip install .'

srun --mpi=pmix --ntasks=8 --container-name=nemo_container --container-mounts=/mswg2/E2E/theo/nemo-cb:/workspace --gpus=8 \
python /workspace/cloudai_nemorun.py --yes --factory llama3_8b log.ckpt.save_last=False data.global_batch_size=32 trainer.val_check_interval=1000 trainer.callbacks="combined_callbacks" trainer.max_steps=100 data.seq_length=4096 trainer.strategy.tensor_model_parallel_size=2 trainer.strategy.context_parallel_size=1
