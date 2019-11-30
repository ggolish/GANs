
import torch
import numpy as np

from tqdm import tqdm


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


def clean_images(images):
    ''' Converts gan form images to normal form '''
    a = 127.5 * np.transpose(images, (0, 2, 3, 1)) + 127.5
    return a.astype('uint8')
