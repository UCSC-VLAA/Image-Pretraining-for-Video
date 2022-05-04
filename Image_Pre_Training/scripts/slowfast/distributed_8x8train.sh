#!/bin/bash
NUM_PROC=$1
PYTHONPATH=$PYTHONPATH:../../ \
shift
python3 -m torch.distributed.launch --master_port=22234 --nproc_per_node=$NUM_PROC train.py "$@" \
[your data path] \
--num-classes 1000 \
--val-split val \
--model v3d_slowfast50_8x8_224 \
--batch-size 256 \
--opt lamb \
--weight-decay 0.02 \
--sched cosine \
--lr 5e-3 \
--lr-cycle-decay 1.0 \
--warmup-epochs 5 --warmup-lr 5e-6 \
--epochs 300 \
--cooldown-epochs 0 \
 --aa rand-m7-mstd0.5-inc1 \
--color-jitter 0 \
--aug-repeats 3 \
--bce-loss \
--reprob 0 \
--mixup 0.1 \
--cutmix 1.0 \
--smoothing 0 \
--train-interpolation bicubic \
--drop-path 0.05 \
--crop-pct 0.95 \
--amp \
--native-amp \
--project slowfast50_8x8_300 \
--workers 8 \
--seed 0 \
--pin-mem  \
--output [your output path] \
--experiment slowfast50_8x8_224_zero_init_a2_300e \
--log-interval 50 \
--log-wandb \

