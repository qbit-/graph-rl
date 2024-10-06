# Learning Heuristic for Treewidth problem
 
This module support Self-Imitation-Learning with A2C


## How to use the code

For more details about functionality please see `arguments.py`

#### Training

Example of training with SIL and initial graph file

```sh
python3 train_ac.py --batch 32  --gcn --log_every 1 --val_every 1  --use_neighbour \
 --use_small  --prob 0.14 --treew 15 --lr 0.1 --hidden 40   --max_shape 96  --hidden 30  --use_sil --use_priority \
 --graph ./graph_test/pace2017_instances/gr/heuristic/he059.gr.xz
```

for training on random graph use flag `--use_erdos` and `--random_graph` as string description erdos or k-tree

#### Evaluation

```sh
python3 evaluate.py --graph_path  ./graph_test/pace2017_instances/gr/heuristic/he075.gr.xz --hidden 30 --use_ne \
 --use_small --gcn --resume ./best_models/model_reinforce.pt  --verbose
```

for using exact solver don't forget add execution rights:

```bash
chmod +x ./thirdparty/pace2017_solvers/tamaki_treewidth/tw-exact
```


### Features

Support training with [Self Imitation Learning](https://arxiv.org/abs/1806.05635) `--use_sil`

Also agent can be trained with additional reward from [curiosity](https://arxiv.org/pdf/1705.05363.pdf) module `--use_intrinsic`  
