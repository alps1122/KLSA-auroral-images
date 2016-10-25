import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf
import numpy as np
import dsift
from scipy import misc
import matplotlib.pyplot as plt
import h5py
siftFeaDim = 128
posParaNum = 4
saveName = 'balance500SIFT.hdf5'
saveFolder = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/Features/'

def calImgDSift(imgFile, gridSize, sizeRange):
    print imgFile
    patches, positions, im = esg.generateGridPatchData(imgFile, gridSize, sizeRange)
    feaVecs = np.zeros((len(patches), siftFeaDim))
    for i in range(len(patches)):
        patchSize = positions[i][-1]
        extractor = dsift.SingleSiftExtractor(patchSize)
        feaVec = extractor.process_image(patches[i])
        feaVecs[i, :] = feaVec
    return feaVecs, np.array(positions)

def calSIFTFeaSet(dataFolder, labelFile, classNum, imgType, gridSize, sizeRange):
    names, labels = plf.parseNL(labelFile)
    auroraData = plf.arrangeToClasses(names, labels, classNum)

    f = h5py.File(saveFolder + saveName, 'w')
    f.attrs['dataFolder'] = dataFolder
    ad = f.create_group('auroraData')
    for c, imgs in auroraData.iteritems():
        ascii_imgs = [n.encode('ascii', 'ignore') for n in imgs]
        ad.create_dataset(c, (len(ascii_imgs),), 'S10', ascii_imgs)

    feaSet = f.create_group('feaSet')
    posSet = f.create_group('posSet')
    for c, imgs in auroraData.iteritems():
        feaArr = np.empty((0, siftFeaDim))
        posArr = np.empty((0, posParaNum))
        for name in imgs:
            imgFile = dataFolder+name+imgType
            feaVec, posVec = calImgDSift(imgFile, gridSize, sizeRange)
            feaArr = np.append(feaArr, feaVec, axis=0)
            posArr = np.append(posArr, posVec, axis=0)
        feaSet.create_dataset(c, feaArr.shape, 'f', feaArr)
        posSet.create_dataset(c, posArr.shape, 'i', posArr)
    f.close()
    print saveFolder+saveName+' saved'
    return 0


if __name__ == '__main__':
    dataFolder = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/labeled2003_38044/'
    labelFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/balanceSampleFrom_one_in_minute.txt'
    names, labels = plf.parseNL(labelFile)
    classNum = 4
    gridSize = [10, 10]
    sizeRange = [10, 30]
    imgType = '.bmp'
    auroraData = plf.arrangeToClasses(names, labels, classNum)
    img = misc.imread(dataFolder + auroraData['1'][0] + imgType)
    print img.shape

    patches, positions, im = esg.generateGridPatchData(dataFolder+auroraData['1'][0]+imgType, gridSize, sizeRange)

    patchSize = positions[110][-1]
    print patchSize

    extractor = dsift.SingleSiftExtractor(patchSize)
    feaVec = extractor.process_image(patches[110])

    feaVecs, pos = calImgDSift(dataFolder+auroraData['2'][400]+imgType, gridSize, sizeRange)

    dataSIFTFeature = calSIFTFeaSet(dataFolder, labelFile, classNum, imgType, gridSize, sizeRange)

    # print pos[110][-1]
    # plt.figure()
    # plt.plot(feaVec[0], 'r')
    # plt.figure()
    # plt.plot(feaVecs[600][0], 'b')
    # plt.show()

    pass