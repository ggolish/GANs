#!/usr/bin/env python3

''' Module for handling GAN training '''

import torch
import sys
import os
import json

from tqdm import tqdm

# Default training parameters
DEFAULT_SETTINGS = {
    'iterations': 1000,
    'sample_interval': 20,
    'learning_rate': 0.0002,
    'batch_size': 128
}

INFO = {
    'iterations': [int, 'Number of iterations to train model.'],
    'sample_interval': [int, 'Interval between metrics collections.'],
    'learning_rate': [float, 'Learning rate for gradient descent.'],
    'batch_size': [int, 'The size of a training batch.']
}


def train(gan, name, dl, settings={}, dest='results', checkpoints=True,
          ci=0, cr={}):
    ''' Trains a GAN on the given data and handles results '''

    # Parse settings
    S = DEFAULT_SETTINGS.copy()
    for key in settings:
        if key in S:
            S[key] = settings[key]
        else:
            sys.stderr.write(f'Warning: Invalid training param {key}!\n')

    # Create destination to store checkpoints/results
    dest = os.path.join(dest, name)
    os.makedirs(dest, exist_ok=True)

    # Store training session settings for later recovery
    settings_dest = os.path.join(dest, 'settings.json')
    with open(settings_dest, 'w') as fd:
        json.dump(S, fd)

    # Run training session and store results
    curr_iteration = ci
    results = cr
    print(f'Training {name}...')
    for metrics in gan.train_no_gp(dl, S['iterations'], ci, S['learning_rate'],
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


def recover_training_state(name, dest='results'):
    ''' Recover a previous training session '''

    # Load checkpoints
    checkpoints = load_checkpoints(name, dest=dest)

    # Bring results up to date
    results = {}
    for c in checkpoints:
        for key in c['metrics']:
            if key not in results:
                results[key] = [c['metrics'][key]]
            else:
                results[key].append(c['metrics'][key])

    # Load training settings
    settings_dest = os.path.join(dest, name, 'settings.json')
    with open(settings_dest, 'r') as fd:
        S = json.load(fd)

    return (S, checkpoints[-1], results)


def is_training_started(name, dest='results'):
    ''' Checks if the training session has been started '''
    path = os.path.join(dest, name)
    return os.path.exists(path)


def store_checkpoint(dest, name, gan, metrics, iteration):
    ''' Stores a copy of the GAN at a checkpoint during training '''
    path = '{}-checkpoint-{:05d}.pt'.format(name, iteration)
    path = os.path.join(dest, path)
    torch.save({
        'iteration': iteration,
        'metrics': metrics,
        'settings': gan.S,
        'd_state_dict': gan.D.state_dict(),
        'g_state_dict': gan.G.state_dict()
    }, path)


def store_results(dest, name, gan, results):
    ''' Stores the results of a GAN after training '''
    path = '{}-results.pt'.format(name)
    path = os.path.join(dest, path)
    torch.save({
        'results': results,
        'settings': gan.S,
        'd_state_dict': gan.D.state_dict(),
        'g_state_dict': gan.G.state_dict()
    }, path)


def load_checkpoints(name, dest='results'):
    ''' Loads all checkpoints in a training session '''
    dest = os.path.join(dest, name)
    prefix = f'{name}-checkpoint'
    paths = [os.path.join(dest, f)
             for f in os.listdir(dest) if f.startswith(prefix)]
    print('Loading checkpoints:')
    dev = torch.device('cpu')
    checkpoints = [torch.load(p, map_location=dev)
                   for p in tqdm(paths, ascii=True)]
    return sorted(checkpoints, key=lambda c: c['iteration'])


def load_results(name, dest='results'):
    ''' Loads the results of a specific training session '''
    path = os.path.join(dest, name, f'{name}-results.pt')
    if not os.path.exists(path):
        sys.stderr.write(f'Unable to load results for session "{name}".\n')
        return None
    return torch.load(path)
