import matplotlib.pyplot as plt
from src.experiments.calCSAccuracy import calSegClsAccuracy

if __name__ == '__main__':
    feaTypes = ['LBP', 'SIFT', 'His']
    wordsNums = [200, 500]#, 800]
    # patchSizes = [8, 16, 24, 32, 40, 48, 56, 64]
    patchSizes = [16, 32, 48, 64]
    mks = [0]

    resultSaveFolder = '../../Data/Results/segmentation/modelFS_seg_FWP_mk0/'
    curves_dic = {}
    for patchSize in patchSizes:
        for wordsNum in wordsNums:
            for feaType in feaTypes:
                for mk in mks:
                    resultFile = resultSaveFolder + 'segmentation_' + feaType + '_w' + str(wordsNum) + '_s' + str(patchSize) + '_mk' + str(
                        mk) + '.txt'

                    key = feaType + '_' + str(wordsNum)
                    if key not in curves_dic:
                        curves_dic[key] = []

                    cls_acc, seg_acc = calSegClsAccuracy(resultFile)
                    curves_dic[key].append(seg_acc.mean())
                    print feaType, wordsNum, patchSize, seg_acc.mean()
    print curves_dic

    fig, ax = plt.subplots()
    plt.xlabel('patch size')
    plt.ylabel('segmentation accuracy')
    keys = curves_dic.keys()
    keys.sort()
    for k in keys:
        v = curves_dic[k]
        # yticks = range(10, 110, 10)
        # ax.set_yticks(yticks)
        # ax.set_ylim([10, 110])
        # ax.set_xlim([58, 42])
        plt.plot(patchSizes, v, label=k)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()