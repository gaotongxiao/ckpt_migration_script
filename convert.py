# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export weights from the source model to the target model.'
    )
    parser.add_argument('from_ckpt', type=str, help='Path to the source model')
    parser.add_argument(
        'from_keys', type=str, help='Path to the key list of the source model')
    parser.add_argument('to_ckpt', type=str, help='Path to the target model')
    parser.add_argument(
        'to_keys', type=str, help='Path to the key list of the target model')
    parser.add_argument('out_ckpt', type=str, help='Path to the final model')
    parser.add_argument(
        '--zero-missing-keys',
        action='store_true',
        help='Zero out all weights of the target model missing in'
        'the key list.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    from_model = torch.load(args.from_ckpt, map_location='cpu')
    to_model = torch.load(args.to_ckpt, map_location='cpu')
    from_keys = open(args.from_keys, 'r').readlines()
    to_keys = open(args.to_keys, 'r').readlines()
    assert len(from_keys) == len(to_keys)

    #from_model_params = from_model['model']
    from_model_params = from_model['state_dict']
    to_model_params = to_model['state_dict']

    mapped_keys = set()
    for i in range(len(from_keys)):
        from_key = from_keys[i].strip()
        to_key = to_keys[i].strip()
        if from_key == '' or to_key == '':
            continue
        mapped_keys.add(to_key)
        if to_model_params[to_key].shape != from_model_params[from_key].shape:
            print(f'Shape mismatch! The shape of {from_key} is'
                  f'{from_model_params[from_key].shape} but {to_key} is'
                  f'{to_model_params[to_key].shape}')
        to_model_params[to_key] = from_model_params[from_key]

    if args.zero_missing_keys:
        for key in to_model_params.keys():
            if key in mapped_keys:
                continue
            print(f'Zeroing out {to_key}...')
            to_model_params[to_key] = torch.zeros_like(to_model_params[to_key])

    torch.save(to_model, args.out_ckpt)


if __name__ == '__main__':
    main()
