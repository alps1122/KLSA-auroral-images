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
from src.localization.klsSegmentation import mapsToLabels, mergePatchAndRegion
import scipy.io as sio
from src.local_feature.adaptiveThreshold import calculateThreshold

def load_mask_mat(filePath):
    mask = sio.loadmat(filePath)['mask']
    for x in xrange(mask.shape[0]):
        mask[x, :] = mask[x, ::-1]
    for y in xrange(mask.shape[1]):
        mask[:, y] = mask[::-1, y]
    mask = mask.T
    return mask

def SCKLS(paras):
    classHeatMap = region_class_heatMap(paras)
    labels = mapsToLabels(classHeatMap)

    paras['specialType'] = labels  # 0: arc, 1: drapery, 2: radial, 3: hot-spot
    maps3, common_labels, F0 = region_special_map(paras, isReturnMaps=True)

    kls, categoryMap, classMap = mergePatchAndRegion(classHeatMap, maps3, labels, 0.5)
    return labels, kls, categoryMap, classMap, classHeatMap

def testSegmentation(feaType, wordsNum, patchSize, mk, paras, labeledDataFolder, resultSaveFolder, classNum=4,
                     wordsFolder='../../Data/Features/', mini=True):
    paras['sizeRange'] = (patchSize, patchSize)
    paras['patchSize'] = np.array([patchSize, patchSize])
    paras['feaType'] = feaType

    if mini is True:
        mini_str = '_mini'
    else:
        mini_str = ''

    if feaType == 'LBP':
        paras['lbp_wordsFile_s1'] = wordsFolder + 'type4_LBPWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['lbp_wordsFile_s2'] = wordsFolder + 'type4_LBPWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['lbp_wordsFile_s3'] = wordsFolder + 'type4_LBPWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['lbp_wordsFile_s4'] = wordsFolder + 'type4_LBPWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
    if feaType == 'SIFT':
        paras['sift_wordsFile_s1'] = wordsFolder + 'type4_SIFTWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['sift_wordsFile_s2'] = wordsFolder + 'type4_SIFTWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['sift_wordsFile_s3'] = wordsFolder + 'type4_SIFTWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['sift_wordsFile_s4'] = wordsFolder + 'type4_SIFTWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
    if feaType == 'His':
        paras['his_wordsFile_s1'] = wordsFolder + 'type4_HisWords_s1_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['his_wordsFile_s2'] = wordsFolder + 'type4_HisWords_s2_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['his_wordsFile_s3'] = wordsFolder + 'type4_HisWords_s3_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
        paras['his_wordsFile_s4'] = wordsFolder + 'type4_HisWords_s4_s' + str(patchSize) + '_b300_w' + str(wordsNum) + mini_str + '.hdf5'
    resultFile = 'segmentation_' + feaType + '_w' + str(wordsNum) + '_s' + str(patchSize) + '_mk' + str(mk) + '.txt'
    f_result = open(resultSaveFolder + resultFile, 'w')
    if mk == 0:
        paras['mk'] = None
    else:
        paras['mk'] = mk
    IoU_accuracy = np.zeros((classNum, ))
    confusionArray_c = np.zeros((classNum, classNum))
    for c in xrange(0, classNum):
        labelImgFolder_c = labeledDataFolder + str(c + 1) + '_selected/'
        labelMaskFolder_c = labeledDataFolder + str(c + 1) + '_mask/'
        imgFiles = os.listdir(labelImgFolder_c)
        for imgName in imgFiles:
            imgFile = labelImgFolder_c + imgName
            imName = imgName[:-4]
            mask_c = load_mask_mat(labelMaskFolder_c + imName + '.mat')

            paras['thresh'] = calculateThreshold(imgFile)
            paras['imgFile'] = imgFile
            im = skimage.io.imread(imgFile)
            if len(im.shape) == 2:
                img = skimage.color.gray2rgb(im)
            paras['img'] = img
            paras['im'] = im
            # imName = imgFile[-20:-4]
            # ----no rotation----
            class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
            labels, kls, categoryMap, classMap, classHeatMap = SCKLS(paras)
            confusionArray_c[c, labels] += 1

            if False:  # show segmentation results
                plt.figure(10)
                plt.imshow(kls, cmap='gray')
                plt.title(class_names[labels + 1] + '_predict')
                plt.axis('off')

                plt.figure(11)
                plt.imshow(mask_c, cmap='gray')
                plt.title(class_names[c + 1] + '_groundTruth')
                plt.axis('off')

                plt.figure(12)
                plt.imshow(im, cmap='gray')
                plt.title('raw image')
                plt.axis('off')
                plt.show()
                mask_c = mask_c.astype(np.int)
                kls = kls.astype(np.int)
                # else:
                #         print 'classification error!'
            intersectionPixelNum = len(np.argwhere((kls * mask_c) > 0))
            unionPixelNum = len(np.argwhere((kls + mask_c) > 0))
            IoU = float(intersectionPixelNum) / float(unionPixelNum)
            print 'IoU:', IoU
            if labels == c:
                IoU_accuracy[c] += IoU
            f_result.write(imgName + ' ' + str(c) + ' ' + str(labels) + ' ' + str(IoU) + '\n')
    f_result.close()
    print confusionArray_c
    accuracy = confusionArray_c / np.sum(confusionArray_c, axis=1).reshape(classNum, 1)
    rightNums = [confusionArray_c[k, k] for k in xrange(classNum)]
    rightNums = np.array(rightNums, dtype='f')
    IoUs = IoU_accuracy / rightNums
    print accuracy
    print rightNums
    print IoUs
    return 0

if __name__ == '__main__':
    paras = {}
    paras['color_space'] = ['rgb']
    paras['ks'] = [30, 50, 100, 150, 200, 250, 300]
    paras['feature_masks'] = [1, 1, 1, 1]
    paras['overlapThresh'] = 0.9
    paras['scoreThresh'] = 0.7
    eraseMapPath = '../../Data/eraseMap.bmp'
    regionModelWeights = '../../Data/region_classification/output/vgg_cnn_m_1024_fast_rcnn_b500_SR_100_440_iter_10000.caffemodel'
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

    gpu_id = 0
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(regionModelPrototxt, regionModelWeights, caffe.TEST)
    paras['net'] = net
    paras['k'] = 60
    paras['minSize'] = 50

    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['th'] = 0.15
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']

    paras['imResize'] = (256, 256)
    paras['imgSize'] = (440, 440)
    paras['nk'] = 1
    resolution = 1
    gridSize = np.array([resolution, resolution])
    paras['resolution'] = resolution
    paras['gridSize'] = gridSize
    paras['withIntensity'] = 'False'
    paras['diffResolution'] = 'False'
    paras['isSave'] = False
    paras['is_rotate'] = False
    paras['sdaePara'] = None

    paras['returnRegionLabels'] = [0]  # 0: special, 1: rest, 2: common
    paras['train'] = False
    is_showProposals = paras['is_showProposals'] = False

    feaTypes = ['LBP', 'SIFT', 'His']
    wordsNums = [200, 500, 800]
    patchSizes = [8, 16, 24, 32, 40, 48, 56, 64]
    mks = [0]

    labeledDataFolder = '../../Data/segmentation_data/'
    resultSaveFolder = '../../Data/Results/segmentation/modelFS_seg_mini_FWP_mk0/'
    for feaType in feaTypes:
        for wordsNum in wordsNums:
            for patchSize in patchSizes:
                for mk in mks:
                    if (feaType=='LBP') and ((wordsNum<500) or ((wordsNum==500) and (patchSize<32))):
                        print feaType, wordsNum, patchSize
                        continue
                    else:
                        testSegmentation(feaType, wordsNum, patchSize, mk, paras, labeledDataFolder, resultSaveFolder)
    # feaTypes = ['LBP']
    # wordsNums = [500]
    # patchSizes = [16]
    # mks = range(2000, 10000, 400)
    #
    # labeledDataFolder = '../../Data/segmentation_data/'
    # resultSaveFolder = '../../Data/Results/segmentation/seg_mk_LBP_s16_w500_noDetection/'
    # for feaType in feaTypes:
    #     for wordsNum in wordsNums:
    #         for patchSize in patchSizes:
    #             for mk in mks:
    #                 testSegmentation(feaType, wordsNum, patchSize, mk, paras, labeledDataFolder, resultSaveFolder)


    # paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s16_b300_w500.hdf5'
    # paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s16_b300_w500.hdf5'
    # paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s16_b300_w500.hdf5'
    # paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s16_b300_w500.hdf5'
    #
    # paras['sift_wordsFile_s1'] = '../../Data/Features/type4_SIFTWords_s1_s16_b300_w200.hdf5'
    # paras['sift_wordsFile_s2'] = '../../Data/Features/type4_SIFTWords_s2_s16_b300_w200.hdf5'
    # paras['sift_wordsFile_s3'] = '../../Data/Features/type4_SIFTWords_s3_s16_b300_w200.hdf5'
    # paras['sift_wordsFile_s4'] = '../../Data/Features/type4_SIFTWords_s4_s16_b300_w200.hdf5'
    #
    # paras['sdae_wordsFile_s1'] = '../../Data/Features/type4_SDAEWords_s1_s28_b500_special_classification.hdf5'
    # paras['sdae_wordsFile_s2'] = '../../Data/Features/type4_SDAEWords_s2_s28_b500_special_classification.hdf5'
    # paras['sdae_wordsFile_s3'] = '../../Data/Features/type4_SDAEWords_s3_s28_b500_special_classification.hdf5'
    # paras['sdae_wordsFile_s4'] = '../../Data/Features/type4_SDAEWords_s4_s28_b500_special_classification.hdf5'
    # im = np.array(imread(imgFile), dtype='f') / 255
    # paras['im'] = im

    # sdaePara = {}
    # sdaePara['weight_d'] = '../../Data/autoEncoder/layer_diff_mean_s16_final.caffemodel'
    # # sdaePara['weight_s'] = '../../Data/autoEncoder/layer_same_mean_s16_final.caffemodel'
    # sdaePara['weight'] = '../../Data/autoEncoder/layer_same_mean_s28_special_final.caffemodel'
    # sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
    # # sdaePara['meanFile'] = '../../Data/patchData_mean_s16.txt'
    # sdaePara['meanFile'] = '../../Data/patchData_mean_s28_special.txt'
    # sdaePara['patchMean'] = False
    # # layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]
    # layerNeuronNum = [28 * 28, 1000, 1000, 500, 64]
    # sdaePara['layerNeuronNum'] = layerNeuronNum
    #
    # paras['sdaePara'] = sdaePara

    # resultsSaveFolder = '../../Data/Results/segmentation/'
    # result_seg = 'result_segmentation.txt'
    # classNum = 4
    # confusionArray_c = np.zeros((classNum, classNum))
    # IoU_accuracy = np.zeros((classNum, ))
    # labelDataFolder = '../../Data/segmentation_data/'
    # imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031222G074652.bmp'

    # f_seg = open(resultsSaveFolder+result_seg, 'w')
    # for c in xrange(0, classNum):
    #     labelImgFolder_c = labelDataFolder + str(c+1) + '_selected/'
    #     labelMaskFolder_c = labelDataFolder + str(c+1) + '_mask/'
    #     imgFiles = os.listdir(labelImgFolder_c)
    #
    #     for imgName in imgFiles:
    #         imgFile = labelImgFolder_c + imgName
    #         img_c = imread(imgFile)
    #         imName = imgName[:-4]
    #         mask_c = load_mask_mat(labelMaskFolder_c + imName + '.mat')
    #
    #         paras['thresh'] = calculateThreshold(imgFile)
    #         paras['imgFile'] = imgFile
    #         im = skimage.io.imread(imgFile)
    #         if len(im.shape) == 2:
    #             img = skimage.color.gray2rgb(im)
    #         paras['img'] = img
    #         paras['im'] = im
    #         # imName = imgFile[-20:-4]
    #         # ----no rotation----
    #         class_names = ['background', 'arc', 'drapery', 'radial', 'hot-spot']
    #         labels, kls, categoryMap, classMap, classHeatMap = SCKLS(paras)
    #         confusionArray_c[c, labels] += 1
    #
    #         if False:  # show segmentation results
    #             plt.figure(10)
    #             plt.imshow(kls, cmap='gray')
    #             plt.title(class_names[labels+1] + '_predict')
    #             plt.axis('off')
    #
    #             plt.figure(11)
    #             plt.imshow(mask_c, cmap='gray')
    #             plt.title(class_names[c+1] + '_groundTruth')
    #             plt.axis('off')
    #
    #             plt.figure(12)
    #             plt.imshow(im, cmap='gray')
    #             plt.title('raw image')
    #             plt.axis('off')
    #             plt.show()
    #             mask_c = mask_c.astype(np.int)
    #             kls = kls.astype(np.int)
    #     # else:
    #     #         print 'classification error!'
    #         intersectionPixelNum = len(np.argwhere((kls * mask_c) > 0))
    #         unionPixelNum = len(np.argwhere((kls + mask_c) > 0))
    #         IoU = float(intersectionPixelNum) / float(unionPixelNum)
    #         print 'IoU:', IoU
    #         if labels == c:
    #             IoU_accuracy[c] += IoU
    #         f_seg.write(imgName + ' ' + str(c) + ' ' + str(labels) + ' ' + str(IoU) + '\n')
    # f_seg.close()
    # print confusionArray_c
    # accuracy = confusionArray_c / np.sum(confusionArray_c, axis=1).reshape(classNum, 1)
    # rightNums = [confusionArray_c[k, k] for k in xrange(classNum)]
    # rightNums = np.array(rightNums, dtype='f')
    # IoUs = IoU_accuracy/rightNums
    # print accuracy
    # print rightNums
    # print IoUs
    #kls_color = np.zeros(img.shape, dtype='uint8')
    #kls_color[:, :, 0][np.where(kls==1)] = 255
    #alpha = 0.2
    #addImg = (kls_color * alpha + img * (1. - alpha)).astype(np.uint8)
    #plt.figure(12)
    #plt.imshow(addImg)
    #plt.title('add image')
    #plt.axis('off')
    #imsave(resultsSaveFolder + imName + '_merge.jpg', addImg)
