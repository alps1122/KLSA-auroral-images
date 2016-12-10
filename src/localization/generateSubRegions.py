import skimage.data
import selectivesearch
import matplotlib.pyplot as plt
import numpy as np

import sys
import argparse
import warnings
import numpy
import skimage.io
import features
import color_space
import selective_search
import src.util.paseLabeledFile as plf
import segment
import src.preprocess.esg as esg

def generate_subRegions(imgFile, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma,
                        radius=220, centers = np.array([219.5, 219.5])):
    img = skimage.io.imread(imgFile)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    F0, n_region = segment.segment_label(img, sigma, k, minSize)

    eraseLabels = set(list(F0[numpy.where(eraseMap == 1)].flatten()))
    region_patch_list = [[] for i in range(n_region)]
    for l in range(n_region):
        if l in eraseLabels:
            region_patch_list[l] = []
        else:
            region_patch_centers = np.argwhere(F0 == l)
            hw = patchSize / 2
            region_patch_gride = np.zeros((region_patch_centers.shape[0], 4))
            region_patch_gride[:, :2] = region_patch_centers - hw
            region_patch_gride[:, 2:] = patchSize
            patch_list = list(region_patch_gride)
            for ll in patch_list:
                if esg.isWithinCircle(ll, centers, radius):
                    if np.random.rand(1, )[0] < region_patch_ratio:
                        region_patch_list[l].append(ll)

    return F0, region_patch_list

def show_region_patch_grid(imgFile, F0, region_patch_list, alpha):
    img = skimage.io.imread(imgFile)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)
    n_region = len(set(list(F0.flatten())))
    eraseLabels = set(list(F0[numpy.where(eraseMap == 1)].flatten()))
    colors = numpy.random.randint(0, 255, (n_region, 3))
    for e in eraseLabels:
        colors[e] = 0
    color_regions = colors[F0]
    result = (color_regions * alpha + img * (1. - alpha)).astype(numpy.uint8)
    for l in region_patch_list:
        if len(l) != 0:
            plf.showGrid(result, l)

    # plt.imshow(result)
    plt.show()

if __name__ == "__main__":
    imgFile = '/home/niuchuang/data/AuroraData/Aurora_img/4/N20031223G120622.jpg'
    k = 100
    minSize = 500
    patchSize = np.array([28, 28])
    region_patch_ratio = 0.2
    sigma = 0.5
    alpha = 0.6

    imSize = 440
    eraseMap = np.zeros((imSize, imSize))
    radius = imSize / 2
    centers = np.array([219.5, 219.5])
    for i in range(imSize):
        for j in range(imSize):
            if np.linalg.norm(np.array([i, j]) - centers) > radius + 5:
                eraseMap[i, j] = 1

    F0, region_patch_list = generate_subRegions(imgFile, gridSize, region_patch_ratio, eraseMap, k, minSize, sigma)

    show_region_patch_grid(imgFile, F0, region_patch_list, alpha)
