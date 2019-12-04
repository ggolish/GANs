
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


def generate_static_images(gan, rows=5, cols=5):
    """ Generating images from the static vectors """
    # load 25 images for 5 x 5 display
    im_size = gan.S['image_size']
    sz = load_static(rows*cols).to(gan.dev)
    with torch.no_grad():
        frame = list()
        frame = gan.G(sz).cpu().numpy()
        frame = clean_images(frame)
        # Each frame of the gif will be a grid of multiple images
        frame = frame[:cols*rows].reshape(rows, cols, im_size, im_size, 3)
        frame = frame.swapaxes(1, 2)
        frame = frame.reshape(im_size * rows, im_size * cols, 3)

    return frame


def clean_images(images):
    ''' Converts gan form images to normal form '''
    a = 127.5 * np.transpose(images, (0, 2, 3, 1)) + 127.5
    return a.astype('uint8')
