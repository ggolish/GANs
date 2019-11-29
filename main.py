#!/usr/bin/env python3
""" This is the main program to run the art gan """

import sys
import argparse
import artgan
import loader

from ganutils import trainer


def train(args):
    args_dict = vars(args)

    if args.recover:
        ts, cp, res = trainer.recover_training_state(args.name)
        dl = loader.load_dataset(args.dataset,
                                 imsize=cp['settings']['image_size'],
                                 batch_size=ts['batch_size'])
        gan = artgan.GAN(cp['settings'])
        gan.D.load_state_dict(cp['d_state_dict'])
        gan.G.load_state_dict(cp['g_state_dict'])
        if args.cuda:
            gan.cuda()
        trainer.train(gan, args.name, dl, ts, ci=cp['iteration'], cr=res)
    else:
        dl = loader.load_dataset(args.dataset, imsize=args.image_size,
                                 batch_size=args.batch_size)

        gansettings = {}
        for key in artgan.DEFAULT_SETTINGS:
            gansettings[key] = args_dict[key]

        gan = artgan.GAN(gansettings)
        if args.cuda:
            gan.cuda()

        trainsettings = {}
        for key in trainer.DEFAULT_SETTINGS:
            trainsettings[key] = args_dict[key]

        trainer.train(gan, args.name, dl, trainsettings)


def parse_args():

    parser = argparse.ArgumentParser('Artgan')
    subparsers = parser.add_subparsers(title='Subcommands')

    train_parser = subparsers.add_parser('train',
                                         help='Utilities for training a new GAN.')
    train_parser.add_argument('dataset',
                              type=str,
                              choices=['ross', 'cifar',
                                       'cubism', 'impressionism'],
                              help='Dataset used during training.')
    train_parser.add_argument('name', type=str,
                              help='Name of new training session.')

    for key, v in trainer.DEFAULT_SETTINGS.items():
        arg = key.replace('_', '-')
        train_parser.add_argument(f'--{arg}',
                                  type=trainer.INFO[key][0],
                                  help=trainer.INFO[key][1],
                                  default=v)

    for key, v in artgan.DEFAULT_SETTINGS.items():
        arg = key.replace('_', '-')
        train_parser.add_argument(f'--{arg}',
                                  type=artgan.INFO[key][0],
                                  help=artgan.INFO[key][1],
                                  default=v)

    train_parser.add_argument('--cuda', action='store_true',
                              help='Run training on gpu.')
    train_parser.add_argument('--recover', action='store_true',
                              help='Attempt to recover training session.')

    train_parser.set_defaults(func=train)

    args = parser.parse_args(sys.argv[1:])

    return args


if __name__ == '__main__':
    args = parse_args()
    args.func(args)
