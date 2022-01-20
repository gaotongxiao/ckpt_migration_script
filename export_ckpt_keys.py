import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export key list from a checkpoint.')
    parser.add_argument('model_path', type=str, help='The path to .pth model')
    parser.add_argument('output', type=str, help='The output key list file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = torch.load(args.model_path)
    with open(args.output, 'w') as f:
        for k in model['state_dict'].keys():
            #for k in model['model'].keys():
            f.write(k + '\n')
