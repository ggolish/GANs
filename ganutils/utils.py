
import torch
import numpy as np


def generate_images(gan, n):
    gan.cpu()
    with torch.no_grad():
        images = []
        for i in range(n):
            z = gan.get_latent_vec(1)
            img = gan.generate_image(z)[0].numpy()
            images.append(img)
        images = np.array(images)
        return clean_images(images)


def clean_images(images):
    ''' Converts gan form images to normal form '''
    a = 127.5 * np.transpose(images, (0, 2, 3, 1)) + 127.5
    return a.astype('uint8')
