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

    imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031221G035901.bmp'
    k = 100
    feature_masks = [1, 1, 1, 1]  # ['size', 'color', 'texture', 'fill']
    out_prefix = ''
    alpha = 0.5  # alpha value for compositing result image with input image

    im = skimage.io.imread(imgFile)
    if len(im.shape) == 2:
        img = skimage.color.gray2rgb(im)

    (R, F, L, L_regions) = selective_search.hierarchical_segmentation(img, k, feature_masks, eraseMap=None)
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

    erase_map = np.zeros((440, 440))
    centers = np.array([219.5, 219.5])
    for i in range(440):
        for j in range(440):
            if np.linalg.norm(np.array([i, j]) - centers) > 220+5:
                erase_map[i, j] = 1
    (R, F, L, L_regions) = selective_search.hierarchical_segmentation(img, k, feature_masks, eraseMap=erase_map)

    # region_img = F[0]
    # for i in range(60,80):
    #     grid = [L[i]]
    #     sub_regions = L_regions[i]
    #     im_psudo = np.zeros((440, 440))
    #     for l in sub_regions:
    #         im_psudo[np.where(region_img == l)] = 255
    #     plf.showGrid(im_psudo, grid)
    # plt.show()

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
    ks = [50, 100, 200]
    region_set = selective_search.selective_search(img, color_spaces=color_space, ks=ks, feature_masks=feature_masks, eraseMap=erase_map)

    # region_labels = []
    # for i in range(len(region_set)):
    #     region_labels.append(region_set[i][-1])
    #     region_set[i] = region_set[i][0]

    # grids = [x[1] for x in regions]
    proposal_minSize = 100 * 100
    proposal_maxSize = 440 * 220

    # regions = []
    for rs in region_set:
        ri = rs[0]
        I = rs[1]
        eraseLabels = set(list(I[numpy.where(erase_map == 1)].flatten()))
        regions = []
        for r in ri:
            exist_eraseLabels = [l for l in eraseLabels if l in r[2]]
            if len(exist_eraseLabels) == 0:
                grid_axes = r[1]
                h = grid_axes[2] - grid_axes[0]
                w = grid_axes[3] - grid_axes[1]
                if (h*w >= proposal_minSize) and (h*w <= proposal_maxSize):
                    regions.append(r)

        # regions = [x[0] for x in region_set]
        # region_imgs = [x[-1] for x in region_set]

        # proposals = sorted(regions[0])
        # region_img = region_imgs[0]
        proposals = regions
        region_img = I

        for i in range(len(regions)):
            cornors = proposals[i][1]
            grid = [[cornors[0], cornors[1], cornors[2] - cornors[0] + 1, cornors[3] - cornors[1] + 1]]
            sub_regions = proposals[i][-1]
            im_psudo = np.zeros((440, 440))
            for l in sub_regions:
                im_psudo[np.where(region_img==l)] = im[np.where(region_img==l)]
            plf.showGrid(im_psudo, grid)

        # plt.figure(2)

        # plt.imshow(map, cmap='gray')
        plt.show()
    pass