#!/usr/bin/env bash

python3 train_ac.py --batch 32  --gcn --log_every 1 --val_every 1  --use_neighbour  --use_small  --prob 0.14 --treew 15 \
 --lr 0.1 --hidden 40   --max_shape 96  --hidden 40   --use_priority --use_erdos \
 --prob 0.07 --max_shape 100 --use_intrinsic --use_sil