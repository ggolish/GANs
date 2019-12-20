#!/usr/bin/env python3
""" This is the main program to run the art gan """

import sys
import argparse
import artgan
import loader
import ganutils
import imageio
import torch

from ganutils import trainer, visualize


def train(args):
    args_dict = vars(args)

    if args.recover:
        ts, cp, res = trainer.recover_training_state(args.name)
        dl = loader.load_dataset(
            args.dataset,
            imsize=cp['settings']['image_size'],
            batch_size=ts['batch_size'],
            optimize=args.optimize
        )
        gan = artgan.GAN(cp['settings'])
        gan.D.load_state_dict(cp['d_state_dict'])
        gan.G.load_state_dict(cp['g_state_dict'])
        if args.cuda:
            gan.cuda()
        trainer.train(gan, args.name, dl, ts, ci=cp['iteration'], cr=res)
    else:
        dl = loader.load_dataset(
            args.dataset,
            imsize=args.image_size,
            batch_size=args.batch_size,
            optimize=args.optimize
        )

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


def results(args):

    trainer.summary(args.name)

    try:
        results = trainer.load_results(args.name)
    except Exception as e:
        results = None

    # Plot losses
    if args.losses:
        if not results:
            sys.stderr.write("Error: Can not plot losses, session incomplete.\n")
        else:
            d_losses, g_losses = results['results'].values()
            visualize.plot_losses(d_losses, g_losses)

    # Generate and display images
    if args.generate:
        if not results:
            sys.stderr.write("Error: Unable to generate images, session incomplete.\n")
        else:
            r, c = args.generate
            gan = artgan.gan_from_checkpoint(results)
            title = f'{args.name} Generated Images'
            images = ganutils.generate_images(gan, r * c)
            visualize.images_as_grid(images, r, c, name=title)

    # Generate gif
    if args.gif:
        images = []
        for checkpoint in trainer.load_checkpoints(args.name):
            gan = artgan.gan_from_checkpoint(checkpoint)
            if args.cuda:
                gan.cuda()
            images.append(ganutils.generate_static_images(gan))
        if len(images) == 0:
            sys.stderr.write("Error: Unable to load checkpoints.\n")
        else:
            title = f'{args.name}-static-images.gif'
            imageio.mimsave(title, images, duration=0.1)

    # Move gan to cpu
    if args.move:
        print('moving')
        for path, checkpoint in trainer.move_checkpoints(args.name):
            gan = artgan.gan_from_checkpoint(checkpoint)
            gan.cpu()
            torch.save({
                'results': results['results'],
                'settings': gan.S,
                'd_state_dict': gan.D.state_dict(),
                'g_state_dict': gan.G.state_dict()
            }, path)


def parse_args():

    parser = argparse.ArgumentParser('Artgan')
    subparsers = parser.add_subparsers(title='Subcommands')

    # Create the train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Utilities for training a new GAN.'
    )

    train_parser.add_argument(
        'dataset',
        type=str,
        choices=['ross', 'cifar', 'cubism', 'impressionism'],
        help='Dataset used during training.'
    )
    train_parser.add_argument(
        '-o',
        '--optimize',
        default=False,
        action='store_true',
        help='Use dataset optimizations.'
    )
    train_parser.add_argument(
        'name',
        type=str,
        help='Name of new training session.'
    )

    for key, v in trainer.DEFAULT_SETTINGS.items():
        arg = key.replace('_', '-')
        train_parser.add_argument(
            f'--{arg}',
            type=trainer.INFO[key][0],
            help=trainer.INFO[key][1],
            default=v
        )

    for key, v in artgan.DEFAULT_SETTINGS.items():
        arg = key.replace('_', '-')
        train_parser.add_argument(
            f'--{arg}',
            type=artgan.INFO[key][0],
            help=artgan.INFO[key][1],
            default=v
        )

    train_parser.add_argument(
        '--cuda',
        action='store_true',
        help='Run training on gpu.'
    )
    train_parser.add_argument(
        '--recover',
        action='store_true',
        help='Attempt to recover training session.'
    )

    train_parser.set_defaults(func=train)

    # Create the results subcommand
    results_parser = subparsers.add_parser(
        'results',
        help='Utilities to display training results.'
    )

    results_parser.add_argument(
        'name',
        type=str,
        help='Name of training session to display results for.'
    )
    results_parser.add_argument(
        '--losses',
        action='store_true',
        help='Display the losses of the training session [default].'
    )
    results_parser.add_argument(
        '--generate',
        type=int,
        nargs=2,
        help='Generate and display images from generator in nxm grid.'
    )
    results_parser.add_argument(
        '--gif',
        action='store_true',
        help='Generate a gif with a frame from each checkpoint using our static Z.'
    )
    results_parser.add_argument(
        '--move',
        action='store_true',
        help='Move results to a different device.'
    )
    results_parser.add_argument(
        '--cuda',
        action='store_true',
        help='Move gan to gpu before inference.'
    )

    results_parser.set_defaults(func=results)

    args = parser.parse_args(sys.argv[1:])

    return args


if __name__ == '__main__':
    args = parse_args()
    args.func(args)
