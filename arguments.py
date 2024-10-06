import argparse


def init_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--validate', default=False, action='store_true')

    #  Base experiment config
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--exp_name', default='default', type=str, help='name of the experiments')
    parser.add_argument('--resume_path', default=None, type=str, help='path to the weights of experiment')
    parser.add_argument('--save_path', default='./models_backup', type=str, help='path to save model')

    parser.add_argument('--log_every', default=1, type=int, help='frequency of logging')
    parser.add_argument('--val_every', default=3, type=int, help='frequency of validation')
    parser.add_argument('--num_experiments', default=10, type=int, help='size of batch in buffer')
    parser.add_argument('--n_restarts', default=5, type=int, help='size of batch in buffer')

    #  DL specification
    parser.add_argument('--batch_size', default=2000, type=int, help='size of batch in buffer')
    parser.add_argument('--minibatch_size', default=300, type=int, help='size of batch in buffer')
    parser.add_argument('--max_epoch', default=2000, type=int, help='num of training  h ')
    parser.add_argument('--eval_mode', default=False, action='store_true', help='use agent for evaluating')
    parser.add_argument('--use_parallel', default=False, action='store_true', help='use pool of threads')
    parser.add_argument('--schedule_sizes', default=False, action='store_true', help='use increasing of sizes policy')
    #  Optimization
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--use_swa', default=False, action='store_true', help='use SWA during training')
    parser.add_argument('--swa_freq', default=10, type=int, help='frequency of SWA')
    parser.add_argument('--optimizer_type', default='adam', type=str, help='type of optimization method')

    parser.add_argument('--num_workers', default=2, type=int, help='numbers of threads to process data')
    parser.add_argument('--use_neighbour', default=False, action='store_true')
    parser.add_argument('--use_small_features', default=False, action='store_true', help='use feature size equal 2')
    parser.add_argument('--use_sil', default=False, action='store_true', help='Use Self-Imitation')
    parser.add_argument('--use_priority', default=False, action='store_true',
                        help='Use priority replay buffer for SIL module')
    parser.add_argument('--use_intrinsic', default=False, action='store_true', help='use intrinsic reward')

    #  GraphEnv
    parser.add_argument('--use_erdos', default=False, action='store_true', help='use random erdos graph')
    parser.add_argument('--use_pace', default=False, action='store_true', help='use pace graph')
    parser.add_argument('--proba', default=0.1, type=float, help='probability of the edge')
    parser.add_argument('--treewidth', default=10, type=int, help='treewidth for k-tree')
    parser.add_argument('--random_graph_type', default='erdos', type=str, help='type of random graph')
    parser.add_argument('--heuristic_type', default='min_fill', type=str, help='type of euristic')
    parser.add_argument('--use_exact', default=False, action='store_true', help='use exact solution for evaluation')
    parser.add_argument('--reset_nodes', default=False, action='store_true',
                        help='use graph convolution for representation node')
    #  GCN Parameters
    parser.add_argument('--gcn', default=False, action='store_true',
                        help='use graph convolution for representation node')
    parser.add_argument('--gcn_type', default='GCN', type=str, help='type of graph convolution for representation node')

    parser.add_argument('--embd_size', default=8, type=int, help='size of embedding vectors in gcn')
    parser.add_argument('--hidden_gcn', default=64, type=int, help='hidden dim in gcn')
    parser.add_argument('--cost_function', default='treewidth',
                        help='use one the reward function for graph environment', choices=['treewidth', 'flops'])
    parser.add_argument('--graph_path', default='graph_test/inst_4x5_10_2_58.txt', type=str, help='path to the graph file')
    parser.add_argument('--max_shape', type=int, default=10, help='max shape of the input graph')
    parser.add_argument('--graph_type', default='random', type=str, help='type of graph for testing')
    parser.add_argument('--task', default='treewidth',
                        help='type of the task for graph environment', choices=['treewidth', 'mvc'])

    args = parser.parse_args()

    return args
