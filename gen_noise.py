#Parameters
#----------
# image : ndarray
#     Input image data. Will be converted to float.
# mode : str
#     One of the following strings, selecting the type of noise to add:

#     'gauss'     Gaussian-distributed additive noise.
#     'poisson'   Poisson-distributed noise generated from the data.
#     's&p'       Replaces random pixels with 0 or 1.
#     'speckle'   Multiplicative noise using out = image + n*image,where
#                 n is uniform noise with specified mean & variance.


import numpy as np
import os
import cv2
import math
import array
import matplotlib.pyplot as plt
from utils import define_dims, create_coords, bound_arr
from read_data import read_surface, read_surfaces, write_surface, apply_mask
import turbo_colormap_mpl
import perlin
from scipy.ndimage import gaussian_filter
from PIL import Image

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        nanmask = np.isnan(out)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        out[nanmask] = np.nan
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(~np.isnan(image)))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        strength = 0.05
        rand = strength*(np.random.randn(row,col,ch))
        rand = rand.reshape(row,col,ch)        
        noisy = image + rand
        return noisy


def gen_perlin(arr):
    image_height = 1440
    image_width = 720
    noise = perlin.NoiseUtils(image_height, image_width)
    noise.makeTexture(texture = noise.wood)
    # parameters of tecture found in perlin.py

    noisy_img = noise.img
    noisy_img = normalise(noisy_img)
    #noisy_img = gaussian_filter(noisy_img, sigma=3)
    plt.imshow(noisy_img.T)
    plt.show()

    arr = arr + noisy_img
    return arr


def calculate_mdt(mss, geoid, mask=True):
    mdt = mss - geoid
    if mask:
        return apply_mask(0.25, mdt)
    else:
        return mdt


def normalise(arr):
    arr = (arr - np.nanmin(arr))/(np.nanmax(arr) - np.nanmin(arr))
    return arr



def main():
    # path1 = './data/src/'
    path2 = '../data/res/'
    cmippath = '../cmip5/rcp60/'

    rcp60_mdts = read_surfaces('cmip5_rcp60_mdts_yr5.dat', cmippath, number=3,
                               start=100)
    rcp60_mdts = normalise(rcp60_mdts)
    plt.imshow(np.rot90(rcp60_mdts[0], 1), cmap='turbo')
    plt.show()
    print(np.nanmin(rcp60_mdts), np.nanmax(rcp60_mdts))
    noisy_mdt = noisy("speckle", rcp60_mdts)
    print(np.nanmin(noisy_mdt), np.nanmax(noisy_mdt))
    noisy_mdt = normalise(noisy_mdt)
    print(np.nanmin(noisy_mdt), np.nanmax(noisy_mdt))

    plt.imshow(np.flip(noisy_mdt[0].T, 0), cmap='turbo')
    plt.show()

    # perlin_mdt = gen_perlin(rcp60_mdts)
    # plt.imshow(np.flipud(perlin_mdt[0].T), cmap='turbo')
    # plt.show()

    residual = plt.imread('dip_residual.png')
    plt.imshow(np.flipud(residual), cmap='turbo')
    plt.show()
    print(np.nanmin(residual), np.nanmax(residual))

    #residual = unsqueeze
    residual_mdt = rcp60_mdts[0] + residual.T
    print(np.nanmin(residual_mdt), np.nanmax(residual_mdt))
    residual_mdt = bound_arr(normalise(residual_mdt), 0.25, 0.75)
    print(residual_mdt.shape)
    plt.imshow(np.rot90(residual_mdt, 1), cmap='turbo')
    plt.show()

    # mdt = read_surface('dip_1000_dtu15gtim5do0280_rr0004.dat', path2,
                    #    transpose=True)

    # nemo_mdt = read_surface('orca0083_mdt_12th.dat', path2, transpose=True)
    # nemo_mdt = bound_arr(nemo_mdt, -2, 2)


if __name__ == '__main__':
    main()