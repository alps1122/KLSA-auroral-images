import numpy as np
import src.VisWords.VisWordsAnalysis as vwa
import matplotlib.pyplot as plt
import src.local_feature.extractDSiftFeatures as extSift
import src.util.paseLabeledFile as plf
import h5py
from scipy.misc import imread, imresize
import copy

def calImgPatchLabel(wordsFile, feaVectors):


    fw = h5py.File(wordsFile, 'r')
    w1 = fw.get('/1/words')
    w2 = fw.get('/2/words')
    w1 = np.array(w1)
    w2 = np.array(w2)
    num_words = w1.shape[0]
    patch_num = feaVectors.shape[0]
    dis1 = np.zeros((patch_num, num_words))
    dis2 = np.zeros((patch_num, num_words))

    for v in range(patch_num):
        dis1[v, :] = np.linalg.norm(w1-feaVectors[v], axis=1)
        dis2[v, :] = np.linalg.norm(w2-feaVectors[v], axis=1)

    dis1_min = dis1.min(axis=1)
    dis1_min_idx = dis1.argmin(axis=1)
    dis2_min = dis2.min(axis=1)
    dis2_min_idx = dis2.argmin(axis=1)

    w1_common_idx, w2_common_idx = vwa.calCommonVector(wordsFile)

    labelVoc = np.array(((dis1_min-dis2_min) > 0), dtype='i') # class1: 0, class2: 1, common: 2
    for i in range(patch_num):
        if labelVoc[i] == 0:
            if (w1_common_idx == dis1_min_idx[i]).sum() > 0:
                labelVoc[i] = 2
        if labelVoc[i] == 1:
            if (w2_common_idx == dis2_min_idx[i]).sum() > 0:
                labelVoc[i] = 2

    return labelVoc

def calPatchLabelHierarchy(wordsFile_h1, wordsFile_h2, feaVectors):
    labelVectors_h1 = calImgPatchLabel(wordsFile_h1, feaVectors)
    labelVectors_h2 = calImgPatchLabel(wordsFile_h2, feaVectors)

    fea_c1_idx = np.argwhere(labelVectors_h1 == 0)
    fea_h2_idx = np.argwhere(labelVectors_h1 > 0)
    fea_c1_idx = list(fea_c1_idx.reshape(len(fea_c1_idx)))
    fea_h2_idx = list(fea_h2_idx.reshape(len(fea_h2_idx)))

    labelVectors_h1[fea_h2_idx] = 1
    labelVectors_h2[fea_c1_idx] = 0

    labelVec = labelVectors_h1 + labelVectors_h2
    return labelVec



def showLocalLabel(imgFile, labelVec, posVec, imResize=None):
    im = imread(imgFile)
    if imResize:
        im = imresize(im, imResize)
    # types = [0, 1, 2, 3]
    types = set(labelVec)
    colcors = ['red', 'blue', 'green', 'yellow']
    for t in types:
        fig, ax = plt.subplots(figsize=(12,12))
        ax.imshow(im, aspect='equal', cmap='gray')
        pos_idx = np.argwhere(labelVec==t)
        for i in range(pos_idx.shape[0]):
            patch = posVec[pos_idx[i, 0], :]
            ax.add_patch(
                plt.Rectangle((patch[1], patch[0]),
                              patch[2], patch[3],
                              fill=True, facecolor=colcors[t],
                              alpha=0.5)
            )
        plt.axis('off')
        plt.tight_layout()
        plt.draw()

def filterPos(posVec, labelVec, radius, spaceSize):
    pos_num = posVec.shape[0]
    filtered_idx = []
    poses = []
    for i in range(pos_num):
        poses.append([posVec[i, 0], posVec[i, 1]])

    tranSize = range(-radius*spaceSize, (radius+1)*spaceSize, spaceSize)
    # tranSize.remove(0)
    for i in range(pos_num):
        test_pos = poses[i]
        consist_num = 0
        exist_num = 0
        for h in tranSize:
            test_pos_tran_h = copy.deepcopy(test_pos)
            test_pos_tran_h[0] = test_pos[0] + h
            for w in tranSize:
                test_pos_tran_hw = copy.deepcopy(test_pos_tran_h)
                test_pos_tran_hw[1] = test_pos[1] + w
                if test_pos_tran_hw in poses:
                    exist_num += 1
                    neighbor_idx = poses.index(test_pos_tran_hw)
                    if labelVec[i] == labelVec[neighbor_idx]:
                        consist_num += 1
        if (consist_num - 1) < exist_num/2:
            filtered_idx.append(i)
    filtered_pos = np.delete(posVec, filtered_idx, 0)
    filtered_label = np.delete(labelVec, filtered_idx, 0)
    return filtered_pos, filtered_label

if __name__ == '__main__':
    labelFile = '../../Data/balanceSampleFrom_one_in_minute.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    # wordsFile_h1 = '../../Data/Features/SIFTWords_h1_256.hdf5'
    # wordsFile_h2 = '../../Data/Features/SIFTWords_h2_256.hdf5'
    wordsFile_h1 = '../../Data/Features/SIFTWords_h1.hdf5'
    wordsFile_h2 = '../../Data/Features/SIFTWords_h2.hdf5'
    gridSize = np.array([10, 10])
    sizeRange = (20, 20)
    imResize = (256, 256)

    [images, labels] = plf.parseNL(labelFile)

    # imgFile = imagesFolder + images[0] + imgType
    imgName = 'N20040101G093612'
    imgFile = imagesFolder + imgName + imgType
    feaVectors, posVectors = extSift.calImgDSift(imgFile, gridSize, sizeRange, imResize=None)

    # labelVectors_h = calImgPatchLabel(wordsFile_h1, feaVectors)

    labelVectors_h = calPatchLabelHierarchy(wordsFile_h1, wordsFile_h2, feaVectors)
    print labelVectors_h.shape, posVectors.shape
    print np.argwhere(labelVectors_h==0).shape

    showLocalLabel(imgFile, labelVectors_h, posVectors, imResize=None)

    filtered_pos, filtered_label = filterPos(posVectors, labelVectors_h, 1, 10)
    showLocalLabel(imgFile, filtered_label, filtered_pos, imResize=None)

    plt.show()

    # gridPatchData, gridList, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
    # plf.showGrid(im, gridList)
    # plt.show()