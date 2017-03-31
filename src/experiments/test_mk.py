import numpy as np
import matplotlib.pyplot as plt
from src.experiments.calCSAccuracy import calSegClsAccuracy

if __name__ == '__main__':
    resultsFolder = '../../Data/Results/segmentation/modelFS_seg_mk_LBP_s16_w500_noDetection/'

    feaTypes = ['LBP']
    patchSizes = [16]
    wordsNums = [500]
    mks = range(0, 10000, 400)
    auroraTypes = ['Arc', 'Drapery', 'Radial', 'Hot-spot']
    classNum = 4
    result_arr = np.zeros((len(mks), classNum))

    for feaType in feaTypes:
        for patchSize in patchSizes:
            for wordsNum in wordsNums:
                for i in range(len(mks)):
                    mk = mks[i]
                    resultFile = resultsFolder + 'segmentation_' + feaType + '_w' + str(wordsNum) + '_s' + str(patchSize) + '_mk' + str(mk) + '.txt'
                    cls_acc, seg_acc = calSegClsAccuracy(resultFile)
                    result_arr[i] = seg_acc
                    # print cls_acc, seg_acc
    # print result_arr
    #

    fig, ax = plt.subplots()
    plt.xlabel('mk')
    plt.ylabel('segmentation accuracy')

    for i in range(classNum):
        plt.plot(mks, result_arr[:, i], label=auroraTypes[i])

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()