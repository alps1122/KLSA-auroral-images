import numpy as np
import src.VisWords.VisWordsAnalysis as vwa
import matplotlib.pyplot as plt
import src.local_feature.extractDSiftFeatures as extSift
import src.util.paseLabeledFile as plf
import h5py
from scipy.misc import imread, imresize, imsave
import copy
import sys
import math

sys.path.insert(0, '../../caffe/python')
import caffe
import src.local_feature.autoencoder as AE
import src.preprocess.esg as esg
import src.local_feature.extractSDAEFeatures as extSDAE
import src.local_feature.extractLBPFeatures as extlbp
import src.local_feature.generateLocalFeatures as glf


def calPatchLabels2(wordsFile, feaVectors, k=11):
    fw = h5py.File(wordsFile, 'r')
    w1 = fw.get('/1/words')
    w2 = fw.get('/2/words')
    w1 = np.array(w1)
    w2 = np.array(w2)
    num_words = w1.shape[0]
    patch_num = feaVectors.shape[0]
    dis1 = np.zeros((patch_num, num_words))
    dis2 = np.zeros((patch_num, num_words))
    label1 = np.zeros(num_words)  # class1: 0, class2: 1, common: 2
    label2 = np.ones(num_words)

    for v in range(patch_num):
        dis1[v, :] = np.linalg.norm(w1 - feaVectors[v], axis=1)
        dis2[v, :] = np.linalg.norm(w2 - feaVectors[v], axis=1)
    dis = np.append(dis1, dis2, axis=1)
    w1_common_idx, w2_common_idx = vwa.calCommonVector(wordsFile)
    w1_common_list = list(w1_common_idx.reshape(len(w1_common_idx)))
    w2_common_list = list(w2_common_idx.reshape(len(w2_common_idx)))
    label1[w1_common_list] = 2
    label2[w2_common_list] = 2
    labels = np.append(label1, label2)

    dis_sort_idx = np.argsort(dis, axis=1)
    dis_min_idx_k = dis_sort_idx[:, :k]

    patchLabels = np.zeros((patch_num, k))
    for i in range(patch_num):
        patchLabels[i, :] = labels[list(dis_min_idx_k[i, :])]

    return patchLabels


def calPatchLabelHierarchy2(wordsFile_h1, wordsFile_h2, feaVectors, k=11):
    labelVectors_h1 = calPatchLabels2(wordsFile_h1, feaVectors, k=k)  # 0: arc, 1: drapery, 2: common12
    labelVectors_h2 = calPatchLabels2(wordsFile_h2, feaVectors, k=k)  # 0: drapery, 1: radial, 2: common23

    labelVec = np.append(labelVectors_h1, labelVectors_h2, axis=1)
    return labelVec


def generateClassHeatMap(patchLabels, posVectors, classNum, patchSize, imgSize=(440, 440),
                         isHierarchy=True, threshold_h1=0.3, threshold_h2=0.3):
    heatMap = np.zeros((classNum, imgSize[0], imgSize[1]))
    k = patchLabels.shape[-1]
    patchNum = patchLabels.shape[0]
    patchScores = np.zeros((patchNum, classNum))
    if isHierarchy:
        k /= 2
        patchScores_h1 = np.zeros((patchNum, 3))
        patchScores_h2 = np.zeros((patchNum, 3))
        patchLabels_h1 = patchLabels[:, :k]
        patchLabels_h2 = patchLabels[:, k:]
    resize_posVectors = np.zeros(posVectors.shape)
    resize_posVectors[:, -2:] = patchSize  # patchSize is a np.array, [h, w]
    raw_size_w = posVectors[0, -1]
    raw_size_h = posVectors[0, -2]

    for i in range(patchNum):
        if isHierarchy:
            patchScores_h1[i, :] = np.array(np.histogram(patchLabels_h1[i, :], bins=range(3 + 1))[0], dtype='f') / k
            patchScores_h2[i, :] = np.array(np.histogram(patchLabels_h2[i, :], bins=range(3 + 1))[0], dtype='f') / k
            # patchScores_h1[i][np.argwhere(patchScores_h1[i]<threshold_h1])] = 0

            patchScores[i, 0] = (patchScores_h1[i, 0] > threshold_h1) * patchScores_h1[i, 0]
            patchScores[i, 1:] = (patchScores_h2[i, :] > threshold_h1) * patchScores_h2[i, :]
        else:
            patchScores[i, :] = np.array(np.histogram(patchLabels[i, :], bins=range(classNum + 1))[0], dtype='f') / k

        # resize patches
        resize_posVectors[i, 0] = posVectors[i, 0] + int((raw_size_h - patchSize[0]) / 2)
        resize_posVectors[i, 1] = posVectors[i, 1] + int((raw_size_w - patchSize[1]) / 2)

        # fill patch
        for c in range(classNum):
            x1 = int(resize_posVectors[i, 0])
            x2 = int(resize_posVectors[i, 2]) + x1
            y1 = int(resize_posVectors[i, 1])
            y2 = int(resize_posVectors[i, 3]) + y1
            heatMap[c, x1:x2, y1:y2] = patchScores[i, c]
    return heatMap


def showHeatMaps(imgFile, heat_maps, feaType, k, stepsize, t1, t2,
                 saveFolder='../../Data/Results/', imType='.jpg',
                 classes=['arc', 'drapery', 'radio', 'common'], saveOnly=True,
                 label=''):
    im = np.array(imread(imgFile), dtype='f') / 255
    im_name = imgFile[-20:-4]  # N20031223G125641.bmp
    maps_num = heat_maps.shape[0]
    map_h = heat_maps.shape[1]
    map_w = heat_maps.shape[2]

    if saveOnly:
        print 'save only'
        savePath = saveFolder + im_name + '_' + feaType + '_k' + str(k) + '_r' + str(stepsize) + '_t1_' + str(
            t1) + '_t2_' + str(t2) + imType
        max_width = 3
        max_height = int(math.ceil(float(maps_num) / float(max_width)))
        print 'max_width: ', max_width
        print 'max_height ', max_height

        saveImg = np.zeros((map_h*max_height, map_w*max_width))
        for i in range(max_height):
            for j in range(max_width):

                # if i*max_width+j == maps_num:
                #
                #     print 'break1'
                #     break
                # else:
                saveImg[(i * map_h):((i + 1) * map_h), (j * map_w):((j + 1) * map_w)] = heat_maps[i*max_width+j]
                print i, j
                if i * max_width + j == maps_num-1:
                    print 'break1'
                    break
            if i*max_width+j == maps_num-1:
                print 'break2'
                break
        # print saveImg[100:200, 100:200]
        # print im.shape
        # print saveImg[(i * map_h):((i + 1) * map_h), (j + 1 * map_w):((j + 2) * map_w)].shape
        saveImg[(i * map_h):((i + 1) * map_h), (j+1 * map_w):((j + 2) * map_w)] = im
        # print saveImg[100:200, 100:200]
        imsave(savePath, saveImg)
        print savePath + ' saved'
    else:
        print ' not save only'
        _, axes = plt.subplots(2, 3, figsize=(21, 14))
        for i in range(maps_num):
            imSavePath = saveFolder + im_name + '_' + classes[i] + '_' + feaType + '_k' + str(k) + '_r' + str(
                stepsize) + '_t1_' + str(t1) + '_t2_' + str(t2) + imType
            imsave(imSavePath, heat_maps[i])
            if i <= 2:
                axes[0][i].imshow(heat_maps[i], cmap='gray')
                axes[0][i].set_title(classes[i] + ' map')
            else:
                axes[1][i - 3].imshow(heat_maps[i], cmap='gray')
                axes[1][i - 3].set_title(classes[i] + ' map')

        axes[1][i - 2].imshow(im, cmap='gray')
        axes[1][i - 2].set_title('image_class_' + label)

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i][j].axis('off')

        plt.tight_layout()
        plt.draw()
        plt.savefig(
            saveFolder + im_name + '_' + feaType + '_k' + str(k) + '_r' + str(stepsize) + '_t1_' + str(
                t1) + '_t2_' + str(
                t2) + imType)
    return 0


if __name__ == '__main__':
    labelFile = '../../Data/Alllabel2003_38044.txt'
    testImgs = '../../Data/testImages.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    sdae_wordsFile_h1 = '../../Data/Features/SDAEWords_h1.hdf5'
    sdae_wordsFile_h2 = '../../Data/Features/SDAEWords_h2.hdf5'
    sift_wordsFile_h1 = '../../Data/Features/SIFTWords_h1.hdf5'
    sift_wordsFile_h2 = '../../Data/Features/SIFTWords_h2.hdf5'
    lbp_wordsFile_h1 = '../../Data/Features/LBPWords_h1.hdf5'
    lbp_wordsFile_h2 = '../../Data/Features/LBPWords_h2.hdf5'

    sizeRange = (28, 28)
    imResize = (256, 256)
    imgSize = (440, 440)

    [images, labels] = plf.parseNL(labelFile)

    testNames = plf.parseNL(testImgs)

    for imgName in testNames:
        ll = labels[images.index(imgName)]
        for nk in range(1, 21, 2):
            for resolution in range(1, 6):
                # imgFile = imagesFolder + images[0] + imgType
                # imgName = 'N20031226G033831'
                imgFile = imagesFolder + imgName + imgType
                # nk = 1
                th1 = 0.3
                th2 = 0.3
                # resolution = 1
                gridSize = np.array([resolution, resolution])

                # ----------------show sift------------------
                feaVectors, posVectors = glf.genImgLocalFeas(imgFile, 'SIFT', gridSize, sizeRange, imResize=None)

                # show single hierarchy
                # patchLabels = calPatchLabels2(sift_wordsFile_h1, feaVectors, k=11)
                # heat_maps = generateClassHeatMap(patchLabels, posVectors, 3, gridSize, isHierarchy=False)
                # showHeatMaps(imgFile, heat_maps)

                patchLabels = calPatchLabelHierarchy2(sift_wordsFile_h1, sift_wordsFile_h2, feaVectors, k=nk)
                heat_maps_sift = generateClassHeatMap(patchLabels, posVectors, 4, gridSize, isHierarchy=True,
                                                      threshold_h1=th1, threshold_h2=th2)
                # heat_maps_sift = np.random.random((4,440,400))
                showHeatMaps(imgFile, heat_maps_sift, 'SIFT', nk, resolution, th1, th2, label=ll)

                # ----------------show lbp------------------
                feaVectors, posVectors = glf.genImgLocalFeas(imgFile, 'LBP', gridSize, sizeRange, imResize=None)

                patchLabels = calPatchLabelHierarchy2(lbp_wordsFile_h1, lbp_wordsFile_h2, feaVectors, k=nk)
                heat_maps_lbp = generateClassHeatMap(patchLabels, posVectors, 4, gridSize, isHierarchy=True,
                                                     threshold_h1=th1, threshold_h2=th2)
                showHeatMaps(imgFile, heat_maps_lbp, 'LBP', nk, resolution, th1, th2, label=ll)

                # ---------------show SDEA local results--------------
                # define SDAE parameters
                sdaePara = {}
                sdaePara['weight'] = '../../Data/autoEncoder/final_0.01.caffemodel'
                sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
                sdaePara['meanFile'] = '../../Data/patchData_mean.txt'
                channels = 1
                layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]
                sdaePara['layerNeuronNum'] = layerNeuronNum
                _, gl, _ = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
                batchSize = len(gl)
                inputShape = (batchSize, channels, 28, 28)
                sdaePara['inputShape'] = inputShape

                feaVectors, posVectors = glf.genImgLocalFeas(imgFile, 'SDAE', gridSize, sizeRange, sdaePara=sdaePara)

                patchLabels = calPatchLabelHierarchy2(sdae_wordsFile_h1, sdae_wordsFile_h2, feaVectors, k=nk)
                heat_maps_sdae = generateClassHeatMap(patchLabels, posVectors, 4, gridSize, isHierarchy=True,
                                                      threshold_h1=th1, threshold_h2=th2)
                showHeatMaps(imgFile, heat_maps_sdae, 'SDAE', nk, resolution, th1, th2, label=ll)

                # plt.show()
