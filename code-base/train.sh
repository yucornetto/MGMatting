#!/usr/bin/env bash
echo Which PYTHON: `which python`
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=4 main.py \
--config=config/MGMatting-DIM.toml