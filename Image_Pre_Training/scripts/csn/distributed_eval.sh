#!/bin/bash
NUM_PROC=$1
PYTHONPATH=$PYTHONPATH:../../ \
shift
python3 -m torch.distributed.launch --master_port=22234 --nproc_per_node=$NUM_PROC train.py "$@" \
[your data path] \
--num-classes 1000 \
--val-split val \
--model v3d_csn50_ir_224 \
--batch-size 256 \
--opt lamb \
--workers 8 \
--drop-path 0.05 \
 --eval --resume [your checkpoint path] \
