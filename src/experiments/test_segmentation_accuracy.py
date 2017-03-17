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

if __name__ == '__main__':
    paras = {}

    paras['color_space'] = ['rgb']
    paras['ks'] = [30, 50, 100, 150, 200, 250, 300]
    paras['feature_masks'] = [1, 1, 1, 1]
    paras['overlapThresh'] = 0.9
    paras['scoreThresh'] = 0.7

    eraseMapPath = '../../Data/eraseMap.bmp'
    regionModelWeights = '../../Data/region_classification/output/vgg_cnn_m_1024_fast_rcnn_b500_iter_10000.caffemodel'
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
    paras['minSize'] = 50
    paras['patchSize'] = np.array([16, 16])
    paras['region_patch_ratio'] = 0.1
    paras['sigma'] = 0.5
    paras['alpha'] = 0.6
    paras['th'] = 0.15
    paras['types'] = ['arc', 'drapery', 'radial', 'hot_spot']
    # paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s16_300_300_300_300.hdf5'
    # paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s16_300_300_300_300.hdf5'
    # paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s16_300_300_300_300.hdf5'
    # paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s16_300_300_300_300.hdf5'
    paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s48_b300_w200.hdf5'
    paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s48_b300_w200.hdf5'
    paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s48_b300_w200.hdf5'
    paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s48_b300_w200.hdf5'

    # paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_s16_b300_intensity.hdf5'
    # paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_s16_b300_intensity.hdf5'
    # paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_s16_b300_intensity.hdf5'
    # paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_s16_b300_intensity.hdf5'
    # paras['lbp_wordsFile_s1'] = '../../Data/Features/type4_LBPWords_s1_diffResolution_b500_intensity.hdf5'
    # paras['lbp_wordsFile_s2'] = '../../Data/Features/type4_LBPWords_s2_diffResolution_b500_intensity.hdf5'
    # paras['lbp_wordsFile_s3'] = '../../Data/Features/type4_LBPWords_s3_diffResolution_b500_intensity.hdf5'
    # paras['lbp_wordsFile_s4'] = '../../Data/Features/type4_LBPWords_s4_diffResolution_b500_intensity.hdf5'

    paras['sift_wordsFile_s1'] = '../../Data/Features/type4_SIFTWords_s1_s32_b300_w200.hdf5'
    paras['sift_wordsFile_s2'] = '../../Data/Features/type4_SIFTWords_s2_s32_b300_w200.hdf5'
    paras['sift_wordsFile_s3'] = '../../Data/Features/type4_SIFTWords_s3_s32_b300_w200.hdf5'
    paras['sift_wordsFile_s4'] = '../../Data/Features/type4_SIFTWords_s4_s32_b300_w200.hdf5'

    # paras['sift_wordsFile_s1'] = '../../Data/Features/type4_SIFTWords_s1_diffResolution_b500_intensity.hdf5'
    # paras['sift_wordsFile_s2'] = '../../Data/Features/type4_SIFTWords_s2_diffResolution_b500_intensity.hdf5'
    # paras['sift_wordsFile_s3'] = '../../Data/Features/type4_SIFTWords_s3_diffResolution_b500_intensity.hdf5'
    # paras['sift_wordsFile_s4'] = '../../Data/Features/type4_SIFTWords_s4_diffResolution_b500_intensity.hdf5'

    paras['sdae_wordsFile_s1'] = '../../Data/Features/type4_SDAEWords_s1_s28_b500_special_classification.hdf5'
    paras['sdae_wordsFile_s2'] = '../../Data/Features/type4_SDAEWords_s2_s28_b500_special_classification.hdf5'
    paras['sdae_wordsFile_s3'] = '../../Data/Features/type4_SDAEWords_s3_s28_b500_special_classification.hdf5'
    paras['sdae_wordsFile_s4'] = '../../Data/Features/type4_SDAEWords_s4_s28_b500_special_classification.hdf5'

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
    # sdaePara['weight_s'] = '../../Data/autoEncoder/layer_same_mean_s16_final.caffemodel'
    sdaePara['weight'] = '../../Data/autoEncoder/layer_same_mean_s28_special_final.caffemodel'
    sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
    # sdaePara['meanFile'] = '../../Data/patchData_mean_s16.txt'
    sdaePara['meanFile'] = '../../Data/patchData_mean_s28_special.txt'
    sdaePara['patchMean'] = False
    # layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]
    layerNeuronNum = [28 * 28, 1000, 1000, 500, 64]
    sdaePara['layerNeuronNum'] = layerNeuronNum

    paras['sdaePara'] = sdaePara

    paras['feaType'] = 'SIFT'
    paras['withIntensity'] = 'False'
    paras['diffResolution'] = 'False'
    paras['isSave'] = False
    paras['is_rotate'] = False

    paras['returnRegionLabels'] = [0]  # 0: special, 1: rest, 2: common
    paras['train'] = False
    is_showProposals = paras['is_showProposals'] = False

    resultsSaveFolder = '../../Data/Results/segmentation/'
    result_seg = 'result_segmentation.txt'
    classNum = 4
    confusionArray_c = np.zeros((classNum, classNum))
    IoU_accuracy = np.zeros((classNum, ))
    labelDataFolder = '../../Data/segmentation_data/'
    # imgFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/N20031222G074652.bmp'

    f_seg = open(resultsSaveFolder+result_seg, 'w')
    for c in xrange(0, classNum):
        labelImgFolder_c = labelDataFolder + str(c+1) + '_selected/'
        labelMaskFolder_c = labelDataFolder + str(c+1) + '_mask/'
        imgFiles = os.listdir(labelImgFolder_c)

        for imgName in imgFiles:
            imgFile = labelImgFolder_c + imgName
            img_c = imread(imgFile)
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
                plt.title(class_names[labels+1] + '_predict')
                plt.axis('off')

                plt.figure(11)
                plt.imshow(mask_c, cmap='gray')
                plt.title(class_names[c+1] + '_groundTruth')
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
            f_seg.write(imgName + ' ' + str(c) + ' ' + str(labels) + ' ' + str(IoU) + '\n')
    f_seg.close()
    print confusionArray_c
    accuracy = confusionArray_c / np.sum(confusionArray_c, axis=1).reshape(classNum, 1)
    rightNums = [confusionArray_c[k, k] for k in xrange(classNum)]
    rightNums = np.array(rightNums, dtype='f')
    IoUs = IoU_accuracy/rightNums
    print accuracy
    print rightNums
    print IoUs
    #kls_color = np.zeros(img.shape, dtype='uint8')
    #kls_color[:, :, 0][np.where(kls==1)] = 255
    #alpha = 0.2
    #addImg = (kls_color * alpha + img * (1. - alpha)).astype(np.uint8)
    #plt.figure(12)
    #plt.imshow(addImg)
    #plt.title('add image')
    #plt.axis('off')
    #imsave(resultsSaveFolder + imName + '_merge.jpg', addImg)
