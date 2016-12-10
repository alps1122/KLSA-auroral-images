import numpy as np
import src.localization.classHeatMap as chm
import src.localization.generateSubRegions as gsr
import src.local_feature.generateLocalFeatures as glf
from scipy.misc import imread, imresize, imsave

if __name__ == '__main__':
    imgFile = '/home/niuchuang/data/AuroraData/Aurora_img/3/N20031221G101131.jpg'
    sdae_wordsFile_h1 = '../../Data/Features/type4_SDAEWords_h1.hdf5'
    sdae_wordsFile_h2 = '../../Data/Features/type4_SDAEWords_h2.hdf5'
    sift_wordsFile_h1 = '../../Data/Features/type4_SIFTWords_h1.hdf5'
    sift_wordsFile_h2 = '../../Data/Features/type4_SIFTWords_h2.hdf5'
    lbp_wordsFile_h1 = '../../Data/Features/type4_LBPWords_h1.hdf5'
    lbp_wordsFile_h2 = '../../Data/Features/type4_LBPWords_h2.hdf5'
    k = 100
    minSize = 500
    patchSize = np.array([28, 28])
    region_patch_ratio = 0.1
    sigma = 0.5
    alpha = 0.6
    isHard = True

    imSize = 440
    eraseMap = np.zeros((imSize, imSize))
    radius = imSize / 2
    centers = np.array([219.5, 219.5])
    for i in range(440):
        for j in range(440):
            if np.linalg.norm(np.array([i, j]) - centers) > 220 + 5:
                eraseMap[i, j] = 1

    F0, region_patch_list = gsr.generate_subRegions(imgFile, patchSize, region_patch_ratio, eraseMap, k, minSize, sigma)

    eraseLabels = set(list(F0[np.where(eraseMap == 1)].flatten()))

    sizeRange = (28, 28)
    imResize = (256, 256)
    imgSize = (440, 440)
    nk = 9
    resolution = 1
    gridSize = np.array([resolution, resolution])
    im = np.array(imread(imgFile), dtype='f') / 255
    th1 = 0.5
    th2 = 0.5
    im_name = imgFile[-20:-4]
    maps2by2 = {}

    # F0_3 = np.tile(F0.reshape(F0.shape[0], F0.shape[1], 1), (1, 1, 3))
    # for ri in range(len(region_patch_list)):
    #     r = region_patch_list[ri]
    #     if len(r) != 0:
    #         feaVectors, posVectors = glf.genImgLocalFeas(imgFile, 'SIFT', gridSize, sizeRange, gridList=r)
    #         labels = chm.calPatchLabels2by2(sift_wordsFile_h1, sift_wordsFile_h2, feaVectors, nk)
    #
    #         for k, v in labels.iteritems():
    #             v = list(v.flatten())
    #             if k not in maps2by2:
    #                 maps2by2[k] = np.zeros((3, F0.shape[0], F0.shape[1]))
    #             maps2by2[k][0][np.where(F0 == ri)] = float(v.count(0)) / float(len(v))
    #             maps2by2[k][1][np.where(F0 == ri)] = float(v.count(1)) / float(len(v))
    #             maps2by2[k][2][np.where(F0 == ri)] = float(v.count(2)) / float(len(v))
    #
    # for c, m in maps2by2.iteritems():
    #     map3 = np.transpose(m, (1, 0, 2)).reshape(440, 440 * 3)
    #     map3 = np.append(map3, im, axis=1)
    #     imsave(im_name + '_SIFT_' + c + '_region' + '.jpg', map3)


    # for ri in range(len(region_patch_list)):
    #     r = region_patch_list[ri]
    #     if len(r) != 0:
    #         feaVectors, posVectors = glf.genImgLocalFeas(imgFile, 'LBP', gridSize, sizeRange, gridList=r)
    #         labels = chm.calPatchLabels2by2(lbp_wordsFile_h1, lbp_wordsFile_h2, feaVectors, nk)
    #
    #         for k, v in labels.iteritems():
    #             v = list(v.flatten())
    #             if k not in maps2by2:
    #                 maps2by2[k] = np.zeros((3, F0.shape[0], F0.shape[1]))
    #             c1 = float(v.count(0)) / float(len(v))
    #             c2 = float(v.count(1)) / float(len(v))
    #             cc = float(v.count(2)) / float(len(v))
    #
    #             if isHard:
    #                 cs = np.array([c1, c2, cc])
    #                 idx = cs.argmax()
    #                 maps2by2[k][idx][np.where(F0 == ri)] = 1
    #             else:
    #                 maps2by2[k][0][np.where(F0 == ri)] = c1
    #                 maps2by2[k][1][np.where(F0 == ri)] = c2
    #                 maps2by2[k][2][np.where(F0 == ri)] = cc
    #
    # for c, m in maps2by2.iteritems():
    #     map3 = np.transpose(m, (1, 0, 2)).reshape(440, 440 * 3)
    #     map3 = np.append(map3, im, axis=1)
    #     imsave(im_name + '_LBP_' + c + '_region' + '.jpg', map3)

    sdaePara = {}
    sdaePara['weight'] = '../../Data/autoEncoder/final_0.01.caffemodel'
    sdaePara['net'] = '../../Data/autoEncoder/test_net.prototxt'
    sdaePara['meanFile'] = '../../Data/patchData_mean.txt'
    channels = 1
    layerNeuronNum = [28 * 28, 2000, 1000, 500, 128]
    sdaePara['layerNeuronNum'] = layerNeuronNum

    for ri in range(len(region_patch_list)):
        r = region_patch_list[ri]
        batchSize = len(r)
        inputShape = (batchSize, channels, 28, 28)
        sdaePara['inputShape'] = inputShape
        if len(r) != 0:
            feaVectors, posVectors = glf.genImgLocalFeas(imgFile, 'SDAE', gridSize, sizeRange, gridList=r, sdaePara=sdaePara)
            labels = chm.calPatchLabels2by2(sdae_wordsFile_h1, sdae_wordsFile_h2, feaVectors, nk)

            for k, v in labels.iteritems():
                v = list(v.flatten())
                if k not in maps2by2:
                    maps2by2[k] = np.zeros((3, F0.shape[0], F0.shape[1]))
                c1 = float(v.count(0)) / float(len(v))
                c2 = float(v.count(1)) / float(len(v))
                cc = float(v.count(2)) / float(len(v))

                if isHard:
                    cs = np.array([c1, c2, cc])
                    idx = cs.argmax()
                    maps2by2[k][idx][np.where(F0 == ri)] = 1
                else:
                    maps2by2[k][0][np.where(F0 == ri)] = c1
                    maps2by2[k][1][np.where(F0 == ri)] = c2
                    maps2by2[k][2][np.where(F0 == ri)] = cc

    for c, m in maps2by2.iteritems():
        map3 = np.transpose(m, (1, 0, 2)).reshape(440, 440 * 3)
        map3 = np.append(map3, im, axis=1)
        imsave(im_name + '_SDAE_' + c + '_region' + '.jpg', map3)