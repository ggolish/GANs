#!/usr/bin/env python3

from tkinter import *
import argparse
import numpy as np
import artgan
import torch
from ganutils.utils import clean_images
from PIL import Image, ImageTk
from ganutils import trainer

def selection(event):
    """ Right now this should just make every image the same as the one you click """
    global z
    global canvas
    global canvas_image
    global images
    global img
    indx = (event.y // 64) * 10 + event.x // 64
    # Store our selected z
    selz = z[indx].reshape(1,100,1,1)
    z = selz
    print(indx, selz.shape)

    # z = torch.cat(100*[selz]).reshape(100,100,1,1)
    # z = torch.ones(100,100,1,1).to(gan.dev) * z[indx]
    # z = z[0].reshape(100,1,1)
    # z = torch.zeros(rows*cols, 100, 1, 1).to(gan.dev) + selz
    z = z.repeat(rows*cols, 1, 1, 1)
    print(indx, z[0,0])
    # z = z.repeat(100,1,1,1)
    # z = torch.randn(100,100,1,1).to(gan.dev)
    with torch.no_grad():
        images = gan.G(z).cpu().numpy()
        images = clean_images(images)
    images = images[:cols*rows].reshape(rows, cols, 64, 64, 3)
    images = images.swapaxes(1, 2)
    images = images.reshape(height * rows, width * cols, 3)
    img = ImageTk.PhotoImage(Image.fromarray(images))
    canvas.itemconfig(canvas_image, image=img)

def refresh(event):
    global z
    global canvas
    global canvas_image
    global images
    global img
    z = torch.randn(cols*rows,100,1,1).to(gan.dev)
    with torch.no_grad():
        images = gan.G(z).cpu().numpy()
        images = clean_images(images)
    images = images[:cols*rows].reshape(rows, cols, 64, 64, 3)
    images = images.swapaxes(1, 2)
    images = images.reshape(height * rows, width * cols, 3)
    img = ImageTk.PhotoImage(Image.fromarray(images))
    canvas.itemconfig(canvas_image, image=img)

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GAN Visualizer')
    parser.add_argument(
        'name',
        type=str,
        help='Completed training session to use for visualization.'
    )

    parser.add_argument(
        '-r'
        '--rows',
        type=int,
        default=5,
        help='Number of rows to display.'
    )

    parser.add_argument(
        '-c'
        '--cols',
        type=int,
        default=5,
        help='Number of columns to display.'
    )

    args = parser.parse_args()
    name = args.name
    # name = 'ross-2'
    rows = 10
    cols = 10
    try:
        results = trainer.load_results(name)
        gan = artgan.GAN(results['settings'])
        gan.D.load_state_dict(results['d_state_dict'])
        gan.G.load_state_dict(results['g_state_dict'])
        gan.cuda()
    except Exception as e:
        print(e)
    with torch.no_grad():
        z = torch.randn(rows * cols,100,1,1).to(gan.dev)
        images = gan.G(z).cpu().numpy()
        images = clean_images(images)
    root = Tk()
    height, width, no_channels = images[0].shape
    canvas = Canvas(root, width=width*cols, height=height*rows)
    images = images[:cols*rows].reshape(rows, cols, 64, 64, 3)
    images = images.swapaxes(1, 2)
    images = images.reshape(height * rows, width * cols, 3)

    img = ImageTk.PhotoImage(Image.fromarray(images))
    canvas_image = canvas.create_image(0, 0, image=img, anchor=NW)
    canvas.bind("<Button-1>", selection)
    canvas.bind("<Button-3>", refresh)
    canvas.pack()
    root.mainloop()
