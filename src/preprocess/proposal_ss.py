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
import selective_search
import src.util.paseLabeledFile as plf

def generate_color_table(R):
    # generate initial color
    colors = numpy.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors

if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('image',            type=str,   help='filename of the image')
    # parser.add_argument('-k', '--k',        type=int,   default=100, help='threshold k for initial segmentation')
    # parser.add_argument('-c', '--color',    nargs=1,    default='rgb', choices=['rgb', 'lab', 'rgi', 'hsv', 'nrgb', 'hue'], help='color space')
    # parser.add_argument('-f', '--feature',  nargs="+",  default=['texture', 'fill'], choices=['size', 'color', 'texture', 'fill'], help='feature for similarity calculation')
    # parser.add_argument('-o', '--output',   type=str,   default='result', help='prefix of resulting images')
    # parser.add_argument('-a', '--alpha',    type=float, default=1.0, help='alpha value for compositing result image with input image')
    # args = parser.parse_args()

    imgFile = '../../Data/labeled2003_38044/N20031223G120622.bmp'
    k = 100
    feature_masks = [1, 1, 1, 1]  # ['size', 'color', 'texture', 'fill']
    out_prefix = ''
    alpha = 0.5  # alpha value for compositing result image with input image

    img = skimage.io.imread(imgFile)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    (R, F, L) = selective_search.hierarchical_segmentation(img, k, feature_masks, eraseMap=None)
    print('result filename: %s_[0000-%04d].png' % (out_prefix, len(F) - 1))

    # suppress warning when saving result images
    warnings.filterwarnings("ignore", category=UserWarning)

    colors = generate_color_table(R)
    for depth, label in enumerate(F):
        result = colors[label]
        result = (result * alpha + img * (1. - alpha)).astype(numpy.uint8)
        fn = "%s_%04d.png" % (out_prefix, depth)
        skimage.io.imsave(fn, result)
        sys.stdout.write('.')
        sys.stdout.flush()

    print('\n')

    map = np.zeros((440, 440))
    centers = np.array([219.5, 219.5])
    for i in range(440):
        for j in range(440):
            if np.linalg.norm(np.array([i, j]) - centers) > 220+5:
                map[i, j] = 1
    (R, F, L) = selective_search.hierarchical_segmentation(img, k, feature_masks, eraseMap=map)
    out_prefix = 'erase'
    print('result filename: %s_[0000-%04d].png' % (out_prefix, len(F) - 1))

    # suppress warning when saving result images
    warnings.filterwarnings("ignore", category=UserWarning)

    colors = generate_color_table(R)
    for depth, label in enumerate(F):
        result = colors[label]
        result = (result * alpha + img * (1. - alpha)).astype(numpy.uint8)
        fn = "%s_%04d.png" % (out_prefix, depth)
        skimage.io.imsave(fn, result)
        sys.stdout.write('.')
        sys.stdout.flush()

    print('\n')

    color_space = ['rgb']
    ks = [100]
    regions = selective_search.selective_search(img, color_spaces=color_space, ks=ks, feature_masks=feature_masks)

    grids = [x[1] for x in regions]
    plf.showGrid(img, grids)

    # plt.figure(2)

    # plt.imshow(map, cmap='gray')
    plt.show()
    pass