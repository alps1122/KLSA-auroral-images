import numpy as np
import sys
sys.path.insert(0, '../../caffe/python')
import h5py
import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf

def makePatchData(labelFile, patchSize, gridSize=np.array([10, 10]), imgType='.bmp',
                  channels=1, savePath='../../Data/one_in_minute_patch_diff_mean.hdf5',
                  same_mean_file = '../.../Data/patchData_mean_s16.txt',
                  imagesFolder='../../Data/labeled2003_38044/', patchMean=True,
                  saveList='../../Data/patchList_diff_mean.txt', subtract_same_mean=False):
    sizeRange = (patchSize, patchSize)
    [images, labels] = plf.parseNL(labelFile)
    # arragedImages = plf.arrangeToClasses(images, labels, classNum)

    f = h5py.File(savePath, 'w')
    data = f.create_dataset('data', (0, channels, patchSize, patchSize), dtype='f', maxshape=(None, channels, patchSize, patchSize))
    label = f.create_dataset('label', (0, ), dtype='i', maxshape=(None, ))

    if subtract_same_mean:
        patches_mean = 0
        for i in range(len(images)):
            imf = imagesFolder + images[i] + imgType

            gridPatchData, gridList, _ = esg.generateGridPatchData(imf, gridSize, sizeRange)

            patchData = np.array(gridPatchData)
            patches_mean += patchData.mean()
        patch_mean = patches_mean / len(images)
        print 'patch number: ' + str(data.shape[0])
        print 'patch mean: ' + str(patch_mean)
        with open(same_mean_file, 'w') as f2:
            f2.write('patch_mean: ' + str(patch_mean))
        f2.close()
    else:
        patch_mean = 0

    print 'patch_mean: ', patch_mean
    for i in range(len(images)):
        imf = imagesFolder + images[i] + imgType
        print imf

        gridPatchData, gridList, _ = esg.generateGridPatchData(imf, gridSize, sizeRange)

        patchData = [p.reshape(channels, patchSize, patchSize) for p in gridPatchData]
        patchData = np.array(patchData) - patch_mean
        if patchMean:
            means = np.mean(np.mean(patchData, axis=-1), axis=-1)
            means = means.reshape(means.shape[0], means.shape[1], 1, 1)
            means = np.tile(means, (1, 1, patchSize, patchSize))
            patchData -= means
        labelData = np.full((len(gridList), ), int(labels[i]), dtype='i')

        oldNum = data.shape[0]
        newNum = oldNum + patchData.shape[0]
        data.resize(newNum, axis=0)
        data[oldNum:newNum, :, :, :] = patchData
        label.resize(newNum, axis=0)
        label[oldNum:newNum, ] = labelData

    f.close()
    print 'make patch data done!'

    with open(saveList, 'w') as f1:
        f1.write(savePath)
    f1.close()
    print saveList + ' saved!'

    return 0

if __name__ == '__main__':

    labelFile = '../../Data/balanceSampleFrom_one_in_minute.txt'
    # labelFile = '../../Data/type4_600_300_300_300.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    # sizeRange = (30, 30)
    patchSize = 16
    savePatch_same_mean_s16 = '../../Data/type4_same_mean_s16.hdf5'
    saveList_same_mean_s16 = '../../Data/type4_same_mean_s16.txt'
    savePatch_diff_mean_s16 = '../../Data/type4_diff_mean_s16.hdf5'
    saveList_diff_mean_s16 = '../../Data/type4_diff_mean_s16.txt'

    makePatchData(labelFile, patchSize, patchMean=False, subtract_same_mean=True,
                  savePath=savePatch_same_mean_s16, saveList=saveList_same_mean_s16)
    makePatchData(labelFile, patchSize, patchMean=True, subtract_same_mean=False,
                  savePath=savePatch_diff_mean_s16, saveList=saveList_diff_mean_s16)

    # [images, labels] = plf.parseNL(labelFile)
    #
    # imgFile = imagesFolder + images[0] + imgType
    # gridPatchData, gridList, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
    #
    # print gridList[0:5]
    # plf.showGrid(im, gridList[0:5])
    # plt.show()