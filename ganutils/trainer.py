#!/usr/bin/env python3

''' Module for handling GAN training '''

import torch
import sys
import os

# Default training parameters
DEFAULT_SETTINGS = {
    'iterations': 1000,
    'sample_interval': 20,
    'learning_rate': 0.0002,
    'batch_size': 128,
    'dest': 'results'
}


def train(gan, name, dl, settings={}, checkpoints=True):
    ''' Trains a GAN on the given data and handles results '''

    # Parse settings
    S = DEFAULT_SETTINGS.copy()
    for key in settings:
        if key in S:
            S[key] = settings[key]
        else:
            sys.stderr.write(f'Warning: Invalid training param {key}!\n')

    # Create destination to store checkpoints/results
    dest = os.path.join(S['dest'], name)
    os.makedirs(dest, exist_ok=True)

    # Run training session and store results
    curr_iteration = 0
    results = {}
    for metrics in gan.train_no_gp(dl, S['iterations'], S['learning_rate'],
                                   S['sample_interval'], S['batch_size']):
        for key in metrics:
            if key not in results:
                results[key] = [metrics[key]]
            else:
                results[key].append(metrics[key])
        if checkpoints:
            store_checkpoint(dest, name, gan, metrics, curr_iteration)
        curr_iteration += S['sample_interval']

    store_results(dest, name, gan, results)

    return results


def store_checkpoint(dest, name, gan, metrics, iteration):
    ''' Stores a copy of the GAN at a checkpoint during training '''
    path = '{}-checkpoint-{:05d}.pt'.format(name, iteration)
    path = os.path.join(dest, path)
    torch.save({
        'iteration': iteration,
        'metrics': metrics,
        'd_state_dict': gan.D.state_dict(),
        'g_state_dict': gan.G.state_dict()
    }, path)


def store_results(dest, name, gan, results):
    ''' Stores the results of a GAN after training '''
    path = '{}-results.pt'.format(name)
    path = os.path.join(dest, path)
    torch.save({
        'results': results,
        'd_state_dict': gan.D.state_dict(),
        'g_state_dict': gan.G.state_dict()
    }, path)
