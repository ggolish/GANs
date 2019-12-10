#!/usr/bin/env python3

from tkinter import *
import numpy as np
import artgan
import torch
from ganutils.utils import clean_images
from PIL import Image, ImageTk
from ganutils import trainer

def selection(event):
    global z
    global canvas
    global canvas_image
    global images
    global img
    indx = (event.y // 64) * 10 + event.x // 64
    # Store our selected z
    # selz = z[indx]
    # z = torch.cat(100*[selz]).reshape(100,100,1,1)
    # z = torch.ones(100,100,1,1).to(gan.dev) * z[indx]
    z = torch.randn(100,100,1,1).to(gan.dev)
    with torch.no_grad():
        images = gan.G(z).cpu().numpy()
        images = clean_images(images)
    images = images.reshape(10, 10, 64, 64, 3)
    images = images.swapaxes(1, 2)
    images = images.reshape(height * 10, width * 10, 3)
    img = ImageTk.PhotoImage(Image.fromarray(images))
    canvas.itemconfig(canvas_image, image=img)





if __name__ == '__main__':
    name = 'ross-2'
    try:
        results = trainer.load_results(name)
        gan = artgan.GAN(results['settings'])
        gan.D.load_state_dict(results['d_state_dict'])
        gan.G.load_state_dict(results['g_state_dict'])
        gan.cuda()
    except Exception as e:
        print(e)
    with torch.no_grad():
        z = torch.randn(100,100,1,1).to(gan.dev)
        images = gan.G(z).cpu().numpy()
        images = clean_images(images)
    rows = 10
    cols = 10
    root = Tk()
    height, width, no_channels = images[0].shape
    canvas = Canvas(root, width=width*cols, height=height*rows)
    images = images[:cols*rows].reshape(rows, cols, 64, 64, 3)
    images = images.swapaxes(1, 2)
    images = images.reshape(height * rows, width * cols, 3)

    img = ImageTk.PhotoImage(Image.fromarray(images))
    canvas_image = canvas.create_image(0, 0, image=img, anchor=NW)
    canvas.bind("<Button-1>", selection)
    canvas.pack()
    root.mainloop()
