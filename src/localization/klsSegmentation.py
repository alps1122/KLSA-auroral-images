from src.localization.region_category_map import region_special_map, showSelectRegions, showMaps3
from src.localization.regions_classes_map import region_class_heatMap, visRegionClassHeatMap
import sys
sys.path.insert(0, '../../fast-rcnn/caffe-fast-rcnn/python')
import caffe
from scipy.misc import imread, imsave
import skimage.io
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage.transform import rotate
import math
import matplotlib.pyplot as plt

def angleFromCommon(F0, commonLabels):
    coordinates_special = np.zeros((0, 2))
    for ls in commonLabels:
        # print 'ls: ' + str(ls)
        coordinates_special = np.vstack([coordinates_special, np.argwhere(F0 == ls)])
    # print 'coordinates_special', coordinates_special
    pca = PCA()
    pca.fit(coordinates_special)

    components = pca.components_
    main_ax = components[0]
    angle = math.atan(main_ax[0] / main_ax[1]) * (180.0 / math.pi)
    return angle

def mapsToLabels(classHeatMaps):
    regionHeatSizes = np.zeros((classHeatMaps.shape[2]-1, ))
    for i in xrange(1, classHeatMaps.shape[2]):
        map_i = classHeatMaps[:, :, i]
        regionHeatSizes[i-1] = len(list(map_i[np.where(map_i != 0)].flatten()))
    label = regionHeatSizes.argmax()
    # maxSize = regionHeatSizes[label]
    # labels = []
    # for j in xrange(len(regionHeatSizes)):
    #     if regionHeatSizes[j] / maxSize > 0.85:
    #         labels.append(j)
    return label

def mergePatchAndRegion(classHeatMaps, categoryHeatMaps, labels, th):
    classHeatMap = classHeatMaps[:, :, labels+1]
    categoryHeatMap = categoryHeatMaps.values()[0][0]
    mergeRusults = (classHeatMap + categoryHeatMap) / 2
    # mergeRusults = classHeatMap * categoryHeatMap
    mergeRusults[np.where(mergeRusults > th)] = 1
    mergeRusults[np.where(mergeRusults < th)] = 0
    return mergeRusults

if __name__ == '__main__':
    paras = {}
    imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031221G065441.bmp'
    paras['imgFile'] = imgFile
    paras['color_space'] = ['rgb']
    paras['ks'] = [50, 100, 150, 200, 250]
    paras['feature_masks'] = [1, 1, 1, 1]
    paras['overlapThresh'] = 0.7
    paras['scoreThresh'] = 0.7

    eraseMapPath = '../../Data/eraseMap.bmp'
    regionModelWeights = '../../Data/region_classification/output/vgg_cnn_m_1024_fast_rcnn_iter_10000.caffemodel'
    regionModelPrototxt = '../../fast-rcnn/models/VGG_CNN_M_1024/test_kls.prototxt'
    proposal_minSize = 100 * 100
    proposal_maxSize = 440 * 220
    paras['regionSizeRange'] = [proposal_minSize, proposal_maxSize]
    if not os.path.exists(eraseMapPath):
        imSize = 440
        eraseMap = np.zeros((imSize, imSize))
        radius = imSize / 2
        centers = np.array([219.5, 219.5])
        for i in range(440):
            for j in range(440):
                if np.linalg.norm(np.array([i, j]) - centers) > 220:
                    eraseMap[i, j] = 1
        imsave(eraseMapPath, eraseMap)
    else:
        eraseMap = imread(eraseMapPath) / 255
    paras['eraseMap'] = eraseMap
    im = skimage.io.imread(imgFile)
    if len(im.shape) == 2:
        img = skimage.color.gray2rgb(im)
    paras['img'] = img
    paras['im'] = im
    gpu_id = 1
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(regionModelPrototxt, regionModelWeights, caffe.TEST)
    paras['net'] = net

    sift_wordsFile_h1 = '../../Data/Features/type4_SIFTWords_h1_s16_600_300_300_300.hdf5'
    sift_wordsFile_h2 = '../../Data/Features/type4_SIFTWords_h2_s16_600_300_300_300.hdf5'
    sdae_wordsFile_h1_d = '../../Data/Features/type4_SDAEWords_h1_diff_mean_s16_600_300_300_300.hdf5'
    sdae_wordsFile_h2_d = '../../Data/Features/type4_SDAEWords_h2_diff_mean_s16_600_300_300_300.hdf5'
    sdae_wordsFile_h1_s = '../../Data/Features/type4_SDAEWords_h1_same_mean_s16_600_300_300_300.hdf5'
    sdae_wordsFile_h2_s = '../../Data/Features/type4_SDAEWords_h2_same_mean_s16_600_300_300_300.hdf5'
    lbp_wordsFile_h1 = '../../Data/Features/type4_LBPWords_h1_s16_600_300_300_300.hdf5'
    lbp_wordsFile_h2 = '../../Data/Features/type4_LBPWords_h2_s16_600_300_300_300.hdf5'
    cascade_wordsFile = '../../Data/Features/type4_cascadeWords_fea4_s16_600_300_300_300_modify.hdf5'
    SIFTFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_s16_600_300_300_300_reduce.hdf5'
    SDAEFeaFile_reduce_d = '../../Data/Features/type4_SDAEFeas_diff_mean_s16_600_300_300_300_reduce_sameRatio.hdf5'
    LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_s16_600_300_300_300_reduce_sameRatio.hdf5'
    SDAEFeaFile_reduce_s = '../../Data/Features/type4_SDAEFeas_same_mean_s16_600_300_300_300_reduce_sameRatio.hdf5'
    paras['SIFTFeaFile_reduce'] = SIFTFeaFile_reduce
    paras['SDAEFeaFile_reduce_d'] = SDAEFeaFile_reduce_d
    paras['SDAEFeaFile_reduce_s'] = SDAEFeaFile_reduce_s
    paras['LBPFeaFile_reduce'] = LBPFeaFile_reduce
    paras['sift_wordsFile_h1'] = sift_wordsFile_h1
    paras['sift_wordsFile_h2'] = sift_wordsFile_h2
    paras['sdae_wordsFile_h1_d'] = sdae_wordsFile_h1_d
    paras['sdae_wordsFile_h2_d'] = sdae_wordsFile_h2_d
    paras['sdae_wordsFile_h1_s'] = sdae_wordsFile_h1_s
    paras['sdae_wordsFile_h2_s'] = sdae_wordsFile_h2_s
    paras['lbp_wordsFile_h1'] = lbp_wordsFile_h1
    paras['lbp_wordsFile_h2'] = lbp_wordsFile_h2
    paras['cascade_wordsFile'] = cascade_wordsFile
    paras['k'] = 60
    paras['minSize'] = 100
    paras['patchSize'] = np.array([16, 16])
    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['th'] = 0.15
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']
    paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s16_300_300_300_300.hdf5'
    paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s16_300_300_300_300.hdf5'
    paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s16_300_300_300_300.hdf5'
    paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s16_300_300_300_300.hdf5'

    paras['sift_wordsFile_s1'] = '../../Data/Features/type4_SIFTWords_s1_s16_300_300_300_300.hdf5'
    paras['sift_wordsFile_s2'] = '../../Data/Features/type4_SIFTWords_s2_s16_300_300_300_300.hdf5'
    paras['sift_wordsFile_s3'] = '../../Data/Features/type4_SIFTWords_s3_s16_300_300_300_300.hdf5'
    paras['sift_wordsFile_s4'] = '../../Data/Features/type4_SIFTWords_s4_s16_300_300_300_300.hdf5'

    paras['sizeRange'] = (16, 16)
    paras['imResize'] = (256, 256)
    paras['imgSize'] = (440, 440)
    paras['nk'] = 1
    resolution = 1
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize
    # im = np.array(imread(imgFile), dtype='f') / 255
    # paras['im'] = im

    sdaePara = {}
    sdaePara['weight_d'] = '../../Data/autoEncoder/layer_diff_mean_s16_final.caffemodel'
    sdaePara['weight_s'] = '../../Data/autoEncoder/layer_same_mean_s16_final.caffemodel'
    sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
    sdaePara['meanFile'] = '../../Data/patchData_mean_s16.txt'
    sdaePara['patchMean'] = False
    # layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]
    layerNeuronNum = [16 * 16, 1000, 1000, 500, 64]
    sdaePara['layerNeuronNum'] = layerNeuronNum

    paras['sdaePara'] = sdaePara

    paras['feaType'] = 'LBP'
    paras['isSave'] = False
    paras['is_rotate'] = False

    paras['returnRegionLabels'] = [0]  # 0: special, 1: rest, 2: common
    paras['train'] = False

    classHeatMap = region_class_heatMap(paras)
    class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
    labels = mapsToLabels(classHeatMap)
    visRegionClassHeatMap(classHeatMap, class_names)

    paras['specialType'] = labels  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
    maps3, common_labels, F0 = region_special_map(paras, isReturnMaps=True)
    showMaps3(maps3)
    # showSelectRegions(F0, common_labels)
    # kls = mergePatchAndRegion(classHeatMap, maps3, labels, paras['th']*0.95)
    kls = mergePatchAndRegion(classHeatMap, maps3, labels, 0.6)
    plt.figure(10)
    plt.imshow(kls, cmap='gray')
    class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
    plt.title(class_names[labels+1])
    plt.show()
    # -----rotate image
    # showMaps3(maps3)
    # angle = angleFromCommon(F0, common_labels)

    # showSelectRegions(F0, common_labels, angle)
    # # F0, region_labels, eraseLabels = region_special_map(paras)
    # # showSelectRegions(F0, region_labels)
    # plt.figure(8)
    # plt.imshow(paras['im'], cmap='gray')
    # plt.figure(9)
    # plt.imshow(paras['img'])
    # paras['im'] = rotate(im, angle, preserve_range=True).astype(np.uint8)
    # paras['img'] = rotate(img, angle, preserve_range=True).astype(np.uint8)
    # print paras['im'].shape
    # print paras['img'].shape
    # plt.figure(6)
    # plt.imshow(paras['im'], cmap='gray')
    # plt.figure(7)
    # plt.imshow(paras['img'])
    # plt.show()
    # classHeatMap = region_class_heatMap(paras)
    #
    # visRegionClassHeatMap(classHeatMap, class_names)
    #
    # maps3, common_labels, F0 = region_special_map(paras, isReturnMaps=True)
    # showSelectRegions(F0, common_labels)
    #
    # showMaps3(maps3)