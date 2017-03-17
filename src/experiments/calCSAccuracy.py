import numpy as np
import matplotlib.pyplot as plt

def parseResultFile(resultFile):
    f = open(resultFile, 'r')
    lines = f.readlines()
    imageNames = []
    trueLabels = []
    predictLabels = []
    IoUs = []
    for l in lines:
        result = l.split()
        imageNames.append(result[0])
        trueLabels.append(int(result[1]))
        predictLabels.append(int(result[2]))
        IoUs.append(float(result[3]))
    return imageNames, trueLabels, predictLabels, IoUs

if __name__ == '__main__':
    resultFile = '/home/ljm/NiuChuang/KLSA-auroral-images/Data/Results/segmentation/b300_LBP_s32_adaptiveTh_w200.txt'
    class_num = 4
    imageNames, trueLabels, predictLabels, IoUs = parseResultFile(resultFile)
    trueLabels_arr = np.array(trueLabels)
    predictLabels_arr = np.array(predictLabels)
    IoUs_arr = np.array(IoUs)

    confusionMtrx = np.zeros((class_num, class_num))
    segmentation_acc_true = np.zeros((class_num,))
    segmentation_acc_all = np.zeros((class_num, ))
    segmentation_acc_sum = np.zeros((class_num, ))

    for i in xrange(len(imageNames)):
        confusionMtrx[trueLabels_arr[i], predictLabels_arr[i]] += 1
        if trueLabels_arr[i] == predictLabels_arr[i]:
            segmentation_acc_sum[trueLabels_arr[i]] += IoUs_arr[i]
        # print segmentation_acc_sum[trueLabels_arr[i]]

    # num_perclass = len(imageNames) / class_num
    # for i in xrange(class_num):
    #     idx_i = np.where(trueLabels_arr == i)
    #     trueLabels_i = trueLabels_arr[idx_i]
    #     predictLabels_i = predictLabels_arr[idx_i]
    #     IoUs_i = IoUs_arr[idx_i]
    #
    #     diffs_i = trueLabels_i - predictLabels_i
    #     true_idx_i = np.where(diffs_i == 0)[0] + (i*num_perclass)
    #
    #     IoUs_true = IoUs_arr[(true_idx_i,)]
    #     # print IoUs_true.shape
    #     # print IoUs_i.shape
    #     print IoUs_true.sum()
    #     segmentation_acc_true[i] = IoUs_true.mean()
    #     segmentation_acc_all[i] = IoUs_i.mean()
    rightNums = np.array([confusionMtrx[x, x] for x in range(class_num)], dtype='f')
    print confusionMtrx
    # print segmentation_acc_true
    # print segmentation_acc_all
    segmentation_acc_true = segmentation_acc_sum / rightNums
    print segmentation_acc_true

    # print imageNames
    # print trueLabels
    # print predictLabels
    # print IoUs