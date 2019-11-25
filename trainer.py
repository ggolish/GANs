#!/usr/bin/env python3

''' Module for handling generic GAN experiments and the results of those experiments '''

import artgan
import numpy as np
import os
import pickle
import torch
import matplotlib.pyplot as plt
import sys

def train(gan: artgan.GAN, name: str, dest:str="results"):
    
    results = {"d_losses": [], "g_losses": [], "images": []}

    try:
        for metrics in gan.train():
            results["d_losses"].append(np.mean(metrics["d_losses"]))
            results["g_losses"].append(metrics["g_loss"])
            results["images"].append(metrics["img"])
    except KeyboardInterrupt:
        print("Storing results.")
        store_results(results, gan, name, dest=dest)
        sys.exit(0)

    store_results(results, gan, name, dest=dest)


def clean_images(imgs):
    a = 127.5 * np.transpose(imgs, (0, 2, 3, 1)) + 127.5
    return a.astype("int16")

def store_results(results: dict, gan: artgan.GAN, name: str, dest:str="results"):
    dest = os.path.join(dest, name)
    gan_dest = os.path.join(dest, "gan.pt")
    results_dest = os.path.join(dest, "results.pickle")
    settings_dest = os.path.join(dest, "settings.pickle")
    os.makedirs(dest, exist_ok=True)

    torch.save({
        'generator_state_dict': gan.G.arch.state_dict(),
        'critic_state_dict': gan.D.arch.state_dict()
    }, gan_dest)

    with open(settings_dest, "wb") as fd:
        pickle.dump(gan.S, fd)

    results["images"] = np.array(results["images"])
    results["images"] = clean_images(results["images"])
    with open(results_dest, "wb") as fd:
        pickle.dump(results, fd)
    
def load_results(name: str, dest:str="results"):
    dest = os.path.join(dest, name)
    gan_dest = os.path.join(dest, "gan.pt")
    results_dest = os.path.join(dest, "results.pickle")
    settings_dest = os.path.join(dest, "settings.pickle")

    with open(settings_dest, "rb") as fd:
        settings = pickle.load(fd)

    with open(results_dest, "rb") as fd:
        results = pickle.load(fd)

    settings["dataset"] = None
    gan = artgan.GAN(settings)
    pt = torch.load(gan_dest)
    gan.D.arch.load_state_dict(pt["critic_state_dict"])
    gan.G.arch.load_state_dict(pt["generator_state_dict"])

    return (results, gan)

def display_images(results: dict, rows: int, cols: int):
    if rows * cols != len(results['images']):
        sys.stderr.write("Error: invalid number of rows and columns.\n")
        sys.stderr.write(f'{len(results["images"])}\n')
        return
    imsize = results['images'].shape[1]
    channels = results['images'].shape[3]
    a = results['images'].reshape(rows, cols, imsize, imsize, channels)
    a = a.swapaxes(1, 2)
    a = a.reshape(rows * imsize, cols * imsize, channels)
    plt.imshow(a.astype('int16'))
    plt.axis('off')
    plt.show()
    
def explore_hyperparam(param: str, values: iter, name:str="gan", settings:dict={}):
    for value in values:
        print(f"Training with {param} = {value}.")
        settings[param] = value
        gan = artgan.GAN(settings)
        curr_name = f"{name}-{param}-{value}"
        train(gan, curr_name)
        
