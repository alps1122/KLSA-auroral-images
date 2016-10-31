import numpy as np
import sys
sys.path.insert(0, '../../caffe/python')
import h5py
import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf

def makePatchData(labelFile, patchSize, gridSize=np.array([10, 10]), imgType='.bmp',
                  channels=1, savePath='../../Data/balance500Patch.hdf5',
                  imagesFolder='../../Data/labeled2003_38044/', subtract_mean=True):
    sizeRange = (patchSize, patchSize)
    [images, labels] = plf.parseNL(labelFile)
    # arragedImages = plf.arrangeToClasses(images, labels, classNum)

    f = h5py.File(savePath, 'w')
    data = f.create_dataset('data', (0, channels, patchSize, patchSize), dtype='f', maxshape=(None, channels, patchSize, patchSize))
    label = f.create_dataset('label', (0, ), dtype='i', maxshape=(None, ))

    if subtract_mean:
        patches_mean = 0
        for i in range(len(images)):
            imf = imagesFolder + images[i] + imgType

            gridPatchData, gridList, _ = esg.generateGridPatchData(imf, gridSize, sizeRange)

            patchData = np.array(gridPatchData)
            patches_mean += patchData.mean()
        patch_mean = patches_mean / len(images)
        print 'patch number: ' + str(data.shape[0])
        print 'patch mean: ' + str(patch_mean)
        with open('../../Data/patchData_mean.txt', 'w') as f2:
            f2.write('patch_mean: ' + str(patch_mean))
        f2.close()
    else:
        patch_mean = 0

    for i in range(len(images)):
        imf = imagesFolder + images[i] + imgType

        gridPatchData, gridList, _ = esg.generateGridPatchData(imf, gridSize, sizeRange)

        patchData = [p.reshape(channels, patchSize, patchSize) for p in gridPatchData]
        patchData = np.array(patchData) - patch_mean
        labelData = np.full((len(gridList), ), int(labels[i]), dtype='i')

        oldNum = data.shape[0]
        newNum = oldNum + patchData.shape[0]
        data.resize(newNum, axis=0)
        data[oldNum:newNum, :, :, :] = patchData
        label.resize(newNum, axis=0)
        label[oldNum:newNum, ] = labelData

    f.close()

    with open('../../Data/patchDataList.txt', 'w') as f1:
        f1.write('/home/ljm/NiuChuang/KLSA-auroral-images/Data/balance500Patch.hdf5')
    f1.close()

    return 0


if __name__ == '__main__':

    labelFile = '../../Data/balanceSampleFrom_one_in_minute.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    # sizeRange = (30, 30)
    patchSize = 28

    makePatchData(labelFile, patchSize)

    # [images, labels] = plf.parseNL(labelFile)
    #
    # imgFile = imagesFolder + images[0] + imgType
    # gridPatchData, gridList, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
    #
    # print gridList[0:5]
    # plf.showGrid(im, gridList[0:5])
    # plt.show()