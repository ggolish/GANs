
import torch
import numpy as np
import imageio
from tqdm import tqdm

if __name__ == 'ganutils.utils':
    from .staticz import load_static
else:
    from staticz import load_static

def generate_images(gan, n):
    ''' Generates n images from GAN gan '''
    gan.cpu()
    with torch.no_grad():
        images = []
        for i in range(n):
            z = gan.get_latent_vec(1)
            img = gan.generate_image(z)[0].numpy()
            images.append(img)
        images = np.array(images)
        return clean_images(images)


def generate_baseline_images(checkpoints):
    ''' Generates an image from the same latent vector from each checkpoint in
        training session '''
    with torch.no_grad():
        z = checkpoints[0].get_latent_vec(1)
        imgs = []
        print('Generating baseline images:')
        for gan in tqdm(checkpoints, ascii=True):
            img = gan.G(z)[0].numpy()
            imgs.append(img)
    imgs = np.array(imgs)
    return clean_images(imgs)


def generate_static_images(name:str, checkpoints, im_size=64, rows=5, cols=5):
    """ Generating images from the static vectors """
    # load 25 images for 5 x 5 display
    z = checkpoints[0].get_latent_vec(1)
    print(z.shape)
    sz = load_static(rows*cols)
    print(sz.shape)
    with torch.no_grad():
        imgs = list()
        print('Generating static images:')
        for gan in tqdm(checkpoints, ascii=True):
            frame = list()
            frame = gan.G(sz).numpy()
            frame = clean_images(frame)
            # Each frame of the gif will be a grid of multiple images
            frame = frame[:cols*rows].reshape(rows, cols, im_size, im_size, 3).swapaxes(1, 2).reshape(im_size * rows, im_size * cols, 3)
            imgs.append(frame)

    imgs = np.array(imgs)
    print(imgs.shape)
    imageio.mimsave(f'{name}.fast.gif', imgs, duration=0.1)
    return imgs


def clean_images(images):
    ''' Converts gan form images to normal form '''
    a = 127.5 * np.transpose(images, (0, 2, 3, 1)) + 127.5
    return a.astype('uint8')
