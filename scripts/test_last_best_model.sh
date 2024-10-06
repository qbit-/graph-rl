#!/bin/bash

for i in 50 150 250 350 450 550 650
do
    PYTHONPATH='./qtree' python3 train_ac.py --max_shape $i --max_epoch 20  --batch 32  --gcn --log_every 1 --hidden 25 --use_ne  --use_erdos --lr 0.1\
    --use_small  --val_every 1 --seed 32 --optim adam --lr 0.1 --prob 0.15 --resume ./best_models/model_reinforce.pt --eval
done
