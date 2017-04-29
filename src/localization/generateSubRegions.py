import skimage.data
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../selective_search_py')
import argparse
import warnings
import numpy
import skimage.io
import features
import color_space
# import selective_search
import src.util.paseLabeledFile as plf
import segment
import src.preprocess.esg as esg
from scipy.misc import imsave

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def generate_subRegions(imgFileOrImg, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma,
                        radius=220, centers = np.array([219.5, 219.5]), thresh=None, isSaveDetection=False, diffResolution=False,
                        returnFilteroutLabels=False):
    if isinstance(imgFileOrImg, str):
        img = skimage.io.imread(imgFileOrImg)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
    else:
        img = imgFileOrImg

    # im = rgb2gray(img)
    # print im.max(), im.min()
    im = img[:, :, 0]
    if thresh is None:
        thresh = 0

    F0, n_region = segment.segment_label(img, sigma, k, minSize)

    eraseLabels = set(list(F0[numpy.where(eraseMap == 1)].flatten()))
    # plt.imshow(eraseMap)
    # plt.show()
    # print eraseLabels
    region_patch_list = [[] for i in range(n_region)]
    filterout_labels = []
    for l in range(n_region):
        if l in eraseLabels:
            region_patch_list[l] = []
        else:
            region_patch_centers = list(np.argwhere(F0 == l))
            if len(region_patch_centers) == 0:
                continue
            region_values = im[np.where(F0 == l)]
            region_mean = region_values.mean()
            # print thresh
            if region_mean < thresh:
                region_patch_list[l] = []
                filterout_labels.append(l)
            else:
                hw = patchSize / 2
                region_patch_gride = np.zeros((len(region_patch_centers), 4))
                if not diffResolution:
                    region_patch_gride[:, :2] = np.array(region_patch_centers) - hw
                    region_patch_gride[:, 2:] = patchSize
                    patch_list = list(region_patch_gride)
                for ii in range(len(region_patch_centers)):
                    if not diffResolution:
                        ll = patch_list[ii]
                    if np.random.rand(1, )[0] < region_patch_ratio:
                        if diffResolution:
                            patchSize = np.array(esg.centerArr2sizeList(region_patch_centers[ii]))
                            ll = np.zeros((1, 4))
                            ll[:, :2] = region_patch_centers[ii]
                            ll[:, 2:] = patchSize
                            ll = list(ll)[0]
                        if esg.isWithinCircle(ll, centers, radius):
                            region_patch_list[l].append(ll)

    if isSaveDetection and (thresh != 0):
        folder = '../../Data/Results/regionDetection/'
        alpha = 0.6
        colors = numpy.random.randint(0, 255, (n_region, 3))
        print eraseLabels
        for e in eraseLabels:
            colors[e] = 0
        color_regions_before = colors[F0]
        result_before = (color_regions_before * alpha + img * (1. - alpha)).astype(numpy.uint8)
        print filterout_labels
        for e in filterout_labels:
            colors[e] = 0
        color_regions_after = colors[F0]
        result_after = (color_regions_after * alpha + img * (1. - alpha)).astype(numpy.uint8)
        imsave(folder+'before.jpg', result_before)
        imsave(folder+'after.jpg', result_after)
    if returnFilteroutLabels:
        return F0, region_patch_list, eraseLabels, filterout_labels
    else:
        return F0, region_patch_list, eraseLabels

def show_region_patch_grid(imgFile, F0, region_patch_list, alpha, eraseMap, saveFolder=None):
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
    if saveFolder is not None:
        imName = imgFile[-20:-4]
        ii = 1
    for l in region_patch_list:
        if len(l) != 0:
            plf.showGrid(result, l)
            if saveFolder is not None:
                plt.savefig(saveFolder+imName+'_subregion'+str(ii)+'.jpg')
                ii += 1

    # plt.imshow(result)
    plt.show()

if __name__ == "__main__":
    imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031222G122001.bmp'
    k = 100
    minSize = 300
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

    F0, region_patch_list, _ = generate_subRegions(imgFile, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma)
    # show_region_patch_grid(imgFile, F0, region_patch_list, alpha, eraseMap, saveFolder='/home/ljm/NiuChuang/KLSA-auroral-images/Data/Results/initial_segmentation/')
