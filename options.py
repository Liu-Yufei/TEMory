import argparse
from random import seed
import os
def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--output_path', type = str, default = '/home/lyf/code/File/experiment')
    parser.add_argument('--root_dir', type = str, default = '/home/lyf/code/File/')
    parser.add_argument('--lr', type = str, default = '[0.0001]*2000', help = 'learning rates for steps(list form)')
    parser.add_argument('--batch_size', type = int, default = 16) # 
    parser.add_argument('--num_segments', type = int, default = 32)
    parser.add_argument('--seed', type = int, default = -1, help = 'random seed (-1 for no manual seed)')
    parser.add_argument('--len_feature', type = int, default = 2048)
    parser.add_argument('--exp_name', type = str, default = 'TEMory_random')
    parser.add_argument('--debug', action = 'store_true')
    

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args
