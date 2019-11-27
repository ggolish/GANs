
import numpy as np


def clean_images(images):
    ''' Converts gan form images to normal form '''
    a = 127.5 * np.transpose(images, (0, 2, 3, 1)) + 127.5
    return a.astype('uint8')
