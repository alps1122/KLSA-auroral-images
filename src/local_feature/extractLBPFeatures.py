from skimage import feature
import src.util.paseLabeledFile as plf
import src.preprocess.esg as esg
import numpy as np
import h5py
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    sizeRange = (28, 28)
    channels = 1
    posParaNum = 4
    P = 16
    R = 2
    B = 10

    weight = '../../Data/autoEncoder/final_0.01.caffemodel'
    net = '../../Data/autoEncoder/test_net.prototxt'
    img_test = dataFolder + 'N20031221G030001.bmp'
    gridPatchData, gridList, im = esg.generateGridPatchData(img_test, gridSize, sizeRange)

    # LBP_img = feature.local_binary_pattern(im, P, R, method='uniform')
    # LBP_var = feature.local_binary_pattern(im, P, R, method='var')

    LBP_img = feature.local_binary_pattern(gridPatchData[300], P, R, method='uniform')
    LBP_var = feature.local_binary_pattern(gridPatchData[300], P, R, method='var')

    # gridPatchData_lbp, gridList_lbp, im_lbp = esg.generateGridPatchData(LBP_img, gridSize, sizeRange, imNorm=False)

    # print np.sum(LBP_img_p==gridPatchData_lbp[150])

    # plt.figure(1)
    # plt.plot(LBP_img_p.flatten(), 'r')
    # plt.plot(gridPatchData_lbp[100].flatten(), 'b')

    # plt.figure(1)
    # gif, (ax1,ax2) = plt.subplots(1, 2)
    # ax1.imshow(LBP_img, cmap='gray')
    # ax2.imshow(LBP_var, cmap='gray')
    # ax1.axis('off')
    # ax2.axis('off')
    # plt.show()

    # print LBP_img.shape
    # print LBP_img
    # print LBP_var.shape
    # print LBP_var

    nan_map = np.isnan(LBP_var)
    nan_pos = np.argwhere(nan_map == True)
    LBP_var[list(nan_pos[:, 0]), list(nan_pos[:, 1])] = 0
    # print LBP_var

    var_sum = np.sum(LBP_var)
    bin_step = var_sum / B
    # print var_sum, bin_step
    var_bins = np.linspace(0, var_sum, num=B+1)

    var_hist, var_bins = np.histogram(LBP_var.flatten(), bins=var_bins)
    # print var_hist
    # print var_bins

    lbp_bin_num = P+2
    lbp_hist, lbp_bins = np.histogram(LBP_img.flatten(), bins=range(lbp_bin_num+1))
    print lbp_hist, len(lbp_hist), sum(lbp_hist)
    print lbp_bins
