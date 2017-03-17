import numpy as np
from sklearn.cluster import KMeans
import h5py

def generateWords(featureH5File, groups, saveFile, wordsNum, feaDim=None):
    f = h5py.File(featureH5File, 'r')
    f_w = h5py.File(saveFile, 'w')
    feaSet = f.get('feaSet')

    for i in range(len(groups)):
        c = groups[i]
        if isinstance(c, str):
            feas = None
            feas = feaSet.get(c)
            feas = np.array(feas, dtype='float64')
            print 'cluctering class ' + c + ' with shape ' + str(feas.shape)

        if isinstance(c, list):
            feas = np.empty((0, feaDim), dtype='float64')
            for cs in c:
                feat = feaSet.get(cs)
                feat = np.array(feat, dtype='float64')
                feas = np.append(feas, feat, axis=0)
                print 'cluctering class ' + cs + ' with shape ' + str(feas.shape)

        kmeans = None
        Kmeans = KMeans(n_clusters=wordsNum, n_jobs=-1)
        Kmeans.fit(feas)
        cluster_centers = Kmeans.cluster_centers_
        inertia = Kmeans.inertia_
        cc = f_w.create_group(str(i+1))
        cc.attrs['inertia'] = inertia
        cc.create_dataset('words', cluster_centers.shape, 'f', cluster_centers)
    print saveFile + ' saved'
    return 0

if __name__ == '__main__':
    # siftFeaFile = '../../Data/Features/type4_SIFTFeatures.hdf5'
    # SDAEFeaFile = '../../Data/Features/type4_SDAEFeas.hdf5'
    # SDAEFeaFile_diff_mean = '../../Data/Features/type4_SDAEFeas_diff_mean.hdf5'
    # LBPFeaFile = '../../Data/Features/type4_LBPFeatures.hdf5'

    # siftFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_reduce.hdf5'
    # SDAEFeaFile_reduce = '../../Data/Features/type4_SDAEFeas_reduce_sameRatio.hdf5'
    # LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_reduce_sameRatio.hdf5'

    SIFTFeaFile_b300_s16 = '../../Data/Features/type4_SIFTFeatures_s16_300_300_300_300.hdf5'
    SIFTFeaFile_b500_intensity = '../../Data/Features/type4_SIFTFeatures_diffResolution_b500_withIntensity.hdf5'
    SIFTFeaFile_b300_s32 = '../../Data/Features/type4_SIFTFeatures_s32_b300.hdf5'
    SIFTFeaFile_b300_s48 = '../../Data/Features/type4_SIFTFeatures_s48_b300.hdf5'
    # SDAEFeaFile = '../../Data/Features/type4_SDAEFeas_diff_mean_s16_600_300_300_300.hdf5'
    # LBPFeaFile = '../../Data/Features/type4_LBPFeatures_s16_600_300_300_300.hdf5'
    # SDAEFeaFile_s = '../../Data/Features/type4_SDAEFeas_same_mean_s16_600_300_300_300.hdf5'
    LBPFeaFile_b300_s16 = '../../Data/Features/type4_LBPFeatures_s16_300_300_300_300.hdf5'
    # LBPFeaFile_b300_intensity = '../../Data/Features/type4_LBPFeatures_s16_b300_intensity.hdf5'
    LBPFeaFile_b500_intensity = '../../Data/Features/type4_LBPFeatures_diffResolution_b500_withIntensity.hdf5'
    LBPFeaFile_b300_s32 = '../../Data/Features/type4_LBPFeatures_s32_b300.hdf5'
    LBPFeaFile_b300_s48 = '../../Data/Features/type4_LBPFeatures_s48_b300.hdf5'

    wordsNum = 500
    wordsNum_all = 1000
    groups = ['1', '2', '3', '4']
    groups_s1 = ['1', ['2', '3', '4']]
    groups_s2 = ['2', ['1', '3', '4']]
    groups_s3 = ['3', ['1', '2', '4']]
    groups_s4 = ['4', ['1', '2', '3']]
    groups_h1 = ['1', ['2', '3', '4']]
    groups_h2 = ['2', '3', '4']
    groups_all = [['1', '2', '3', '4']]
    # f = h5py.File(LBPFeaFile, 'r')
    # for name in f:
    #     print name
    # feaSet = f.get('feaSet')
    # for c in feaSet:
    #     print c + str(feaSet[c].shape)

    saveFolder = '../../Data/Features/'

    # lbp_saveName_s1 = 'type4_LBPWords_s1_s16_300_300_300_300.hdf5'
    # lbp_saveName_s2 = 'type4_LBPWords_s2_s16_300_300_300_300.hdf5'
    # lbp_saveName_s3 = 'type4_LBPWords_s3_s16_300_300_300_300.hdf5'
    # lbp_saveName_s4 = 'type4_LBPWords_s4_s16_300_300_300_300.hdf5'
    # lbp_saveName_s1234 = 'type4_LBPWords_s1234_s16_300_300_300_300.hdf5'
    # generateWords(LBPFeaFile_b300, groups_s1, saveFolder + lbp_saveName_s1, wordsNum, feaDim=54)
    # generateWords(LBPFeaFile_b300, groups_s2, saveFolder + lbp_saveName_s2, wordsNum, feaDim=54)
    # generateWords(LBPFeaFile_b300, groups_s3, saveFolder + lbp_saveName_s3, wordsNum, feaDim=54)
    # generateWords(LBPFeaFile_b300, groups_s4, saveFolder + lbp_saveName_s4, wordsNum, feaDim=54)
    # generateWords(LBPFeaFile_b300, groups, saveFolder + lbp_saveName_s1234, wordsNum, feaDim=54)

    # lbp_saveName_s1 = 'type4_LBPWords_s1_s16_b300_intensity.hdf5'
    # lbp_saveName_s2 = 'type4_LBPWords_s2_s16_b300_intensity.hdf5'
    # lbp_saveName_s3 = 'type4_LBPWords_s3_s16_b300_intensity.hdf5'
    # lbp_saveName_s4 = 'type4_LBPWords_s4_s16_b300_intensity.hdf5'
    # lbp_saveName_s1234 = 'type4_LBPWords_s1234_s16_b300_intensity.hdf5'
    # lbp_saveName_s1 = 'type4_LBPWords_s1_diffResolution_b500_intensity.hdf5'
    # lbp_saveName_s2 = 'type4_LBPWords_s2_diffResolution_b500_intensity.hdf5'
    # lbp_saveName_s3 = 'type4_LBPWords_s3_diffResolution_b500_intensity.hdf5'
    # lbp_saveName_s4 = 'type4_LBPWords_s4_diffResolution_b500_intensity.hdf5'
    # lbp_saveName_s1234 = 'type4_LBPWords_s1234_diffResolution_b500_intensity.hdf5'
    # generateWords(LBPFeaFile_b500_intensity, groups_s1, saveFolder + lbp_saveName_s1, wordsNum, feaDim=57)
    # generateWords(LBPFeaFile_b500_intensity, groups_s2, saveFolder + lbp_saveName_s2, wordsNum, feaDim=57)
    # generateWords(LBPFeaFile_b500_intensity, groups_s3, saveFolder + lbp_saveName_s3, wordsNum, feaDim=57)
    # generateWords(LBPFeaFile_b500_intensity, groups_s4, saveFolder + lbp_saveName_s4, wordsNum, feaDim=57)
    # generateWords(LBPFeaFile_b500_intensity, groups, saveFolder + lbp_saveName_s1234, wordsNum, feaDim=57)

    lbp_16_s1 = 'type4_LBPWords_s1_s16_b300_w200.hdf5'
    lbp_16_s2 = 'type4_LBPWords_s2_s16_b300_w200.hdf5'
    lbp_16_s3 = 'type4_LBPWords_s3_s16_b300_w200.hdf5'
    lbp_16_s4 = 'type4_LBPWords_s4_s16_b300_w200.hdf5'

    generateWords(LBPFeaFile_b300_s16, groups_s1, saveFolder + lbp_16_s1, 200, feaDim=54)
    generateWords(LBPFeaFile_b300_s16, groups_s2, saveFolder + lbp_16_s2, 200, feaDim=54)
    generateWords(LBPFeaFile_b300_s16, groups_s3, saveFolder + lbp_16_s3, 200, feaDim=54)
    generateWords(LBPFeaFile_b300_s16, groups_s4, saveFolder + lbp_16_s4, 200, feaDim=54)

    lbp_32_s1 = 'type4_LBPWords_s1_s32_b300_w200.hdf5'
    lbp_32_s2 = 'type4_LBPWords_s2_s32_b300_w200.hdf5'
    lbp_32_s3 = 'type4_LBPWords_s3_s32_b300_w200.hdf5'
    lbp_32_s4 = 'type4_LBPWords_s4_s32_b300_w200.hdf5'
    # lbp_32_s1234 = 'type4_LBPWords_s1234_s32_b300_w500.hdf5'
    # generateWords(LBPFeaFile_b300_s32, groups_s1, saveFolder + lbp_32_s1, 200, feaDim=54)
    # generateWords(LBPFeaFile_b300_s32, groups_s2, saveFolder + lbp_32_s2, 200, feaDim=54)
    # generateWords(LBPFeaFile_b300_s32, groups_s3, saveFolder + lbp_32_s3, 200, feaDim=54)
    # generateWords(LBPFeaFile_b300_s32, groups_s4, saveFolder + lbp_32_s4, 200, feaDim=54)
    # generateWords(LBPFeaFile_b300_s32, groups, saveFolder + lbp_32_s1234, wordsNum, feaDim=54)

    lbp_48_s1 = 'type4_LBPWords_s1_s48_b300_w200.hdf5'
    lbp_48_s2 = 'type4_LBPWords_s2_s48_b300_w200.hdf5'
    lbp_48_s3 = 'type4_LBPWords_s3_s48_b300_w200.hdf5'
    lbp_48_s4 = 'type4_LBPWords_s4_s48_b300_w200.hdf5'
    # lbp_48_s1234 = 'type4_LBPWords_s1234_s48_b300_w500.hdf5'
    # generateWords(LBPFeaFile_b300_s48, groups_s1, saveFolder + lbp_48_s1, 200, feaDim=54)
    # generateWords(LBPFeaFile_b300_s48, groups_s2, saveFolder + lbp_48_s2, 200, feaDim=54)
    # generateWords(LBPFeaFile_b300_s48, groups_s3, saveFolder + lbp_48_s3, 200, feaDim=54)
    # generateWords(LBPFeaFile_b300_s48, groups_s4, saveFolder + lbp_48_s4, 200, feaDim=54)
    # generateWords(LBPFeaFile_b300_s48, groups, saveFolder + lbp_48_s1234, 500, feaDim=54)

    lbp_32_s1 = 'type4_LBPWords_s1_s32_b300_w500.hdf5'
    lbp_32_s2 = 'type4_LBPWords_s2_s32_b300_w500.hdf5'
    lbp_32_s3 = 'type4_LBPWords_s3_s32_b300_w500.hdf5'
    lbp_32_s4 = 'type4_LBPWords_s4_s32_b300_w500.hdf5'
    # lbp_32_s1234 = 'type4_LBPWords_s1234_s32_b300_w500.hdf5'
    # generateWords(LBPFeaFile_b300_s32, groups_s1, saveFolder + lbp_32_s1, 500, feaDim=54)
    # generateWords(LBPFeaFile_b300_s32, groups_s2, saveFolder + lbp_32_s2, 500, feaDim=54)
    # generateWords(LBPFeaFile_b300_s32, groups_s3, saveFolder + lbp_32_s3, 500, feaDim=54)
    # generateWords(LBPFeaFile_b300_s32, groups_s4, saveFolder + lbp_32_s4, 500, feaDim=54)
    # generateWords(LBPFeaFile_b300_s32, groups, saveFolder + lbp_32_s1234, wordsNum, feaDim=54)

    lbp_48_s1 = 'type4_LBPWords_s1_s48_b300_w500.hdf5'
    lbp_48_s2 = 'type4_LBPWords_s2_s48_b300_w500.hdf5'
    lbp_48_s3 = 'type4_LBPWords_s3_s48_b300_w500.hdf5'
    lbp_48_s4 = 'type4_LBPWords_s4_s48_b300_w500.hdf5'
    # lbp_48_s1234 = 'type4_LBPWords_s1234_s48_b300_w500.hdf5'
    # generateWords(LBPFeaFile_b300_s48, groups_s1, saveFolder + lbp_48_s1, 500, feaDim=54)
    # generateWords(LBPFeaFile_b300_s48, groups_s2, saveFolder + lbp_48_s2, 500, feaDim=54)
    # generateWords(LBPFeaFile_b300_s48, groups_s3, saveFolder + lbp_48_s3, 500, feaDim=54)
    # generateWords(LBPFeaFile_b300_s48, groups_s4, saveFolder + lbp_48_s4, 500, feaDim=54)
    # generateWords(LBPFeaFile_b300_s48, groups, saveFolder + lbp_48_s1234, 500, feaDim=54)

    # sift_saveName_s1 = 'type4_SIFTWords_s1_diffResolution_b500_intensity.hdf5'
    # sift_saveName_s2 = 'type4_SIFTWords_s2_diffResolution_b500_intensity.hdf5'
    # sift_saveName_s3 = 'type4_SIFTWords_s3_diffResolution_b500_intensity.hdf5'
    # sift_saveName_s4 = 'type4_SIFTWords_s4_diffResolution_b500_intensity.hdf5'
    # sift_saveName_s1234 = 'type4_SIFTWords_s1234_diffResolution_b500_intensity.hdf5'
    # generateWords(SIFTFeaFile_b500_intensity, groups_s1, saveFolder + sift_saveName_s1, wordsNum, feaDim=131)
    # generateWords(SIFTFeaFile_b500_intensity, groups_s2, saveFolder + sift_saveName_s2, wordsNum, feaDim=131)
    # generateWords(SIFTFeaFile_b500_intensity, groups_s3, saveFolder + sift_saveName_s3, wordsNum, feaDim=131)
    # generateWords(SIFTFeaFile_b500_intensity, groups_s4, saveFolder + sift_saveName_s4, wordsNum, feaDim=131)
    # generateWords(SIFTFeaFile_b500_intensity, groups, saveFolder + sift_saveName_s1234, wordsNum, feaDim=131)

    sift_16_s1 = 'type4_SIFTWords_s1_s16_b300_w200.hdf5'
    sift_16_s2 = 'type4_SIFTWords_s2_s16_b300_w200.hdf5'
    sift_16_s3 = 'type4_SIFTWords_s3_s16_b300_w200.hdf5'
    sift_16_s4 = 'type4_SIFTWords_s4_s16_b300_w200.hdf5'

    generateWords(SIFTFeaFile_b300_s16, groups_s1, saveFolder + sift_16_s1, 200, feaDim=128)
    generateWords(SIFTFeaFile_b300_s16, groups_s2, saveFolder + sift_16_s2, 200, feaDim=128)
    generateWords(SIFTFeaFile_b300_s16, groups_s3, saveFolder + sift_16_s3, 200, feaDim=128)
    generateWords(SIFTFeaFile_b300_s16, groups_s4, saveFolder + sift_16_s4, 200, feaDim=128)

    sift_32_s1 = 'type4_SIFTWords_s1_s32_b300_w200.hdf5'
    sift_32_s2 = 'type4_SIFTWords_s2_s32_b300_w200.hdf5'
    sift_32_s3 = 'type4_SIFTWords_s3_s32_b300_w200.hdf5'
    sift_32_s4 = 'type4_SIFTWords_s4_s32_b300_w200.hdf5'
    # sift_32_s1234 = 'type4_SIFTWords_s1234_s32_b300_w500.hdf5'
    # generateWords(SIFTFeaFile_b300_s32, groups_s1, saveFolder + sift_32_s1, 200, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s32, groups_s2, saveFolder + sift_32_s2, 200, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s32, groups_s3, saveFolder + sift_32_s3, 200, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s32, groups_s4, saveFolder + sift_32_s4, 200, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s32, groups, saveFolder + sift_32_s1234, wordsNum, feaDim=128)

    sift_48_s1 = 'type4_SIFTWords_s1_s48_b300_w200.hdf5'
    sift_48_s2 = 'type4_SIFTWords_s2_s48_b300_w200.hdf5'
    sift_48_s3 = 'type4_SIFTWords_s3_s48_b300_w200.hdf5'
    sift_48_s4 = 'type4_SIFTWords_s4_s48_b300_w200.hdf5'
    # sift_48_s1234 = 'type4_SIFTWords_s1234_s48_b300_w500.hdf5'
    # generateWords(SIFTFeaFile_b300_s48, groups_s1, saveFolder + sift_48_s1, 200, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s48, groups_s2, saveFolder + sift_48_s2, 200, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s48, groups_s3, saveFolder + sift_48_s3, 200, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s48, groups_s4, saveFolder + sift_48_s4, 200, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s48, groups, saveFolder + sift_48_s1234, 500, feaDim=128)

    sift_32_s1 = 'type4_SIFTWords_s1_s32_b300_w500.hdf5'
    sift_32_s2 = 'type4_SIFTWords_s2_s32_b300_w500.hdf5'
    sift_32_s3 = 'type4_SIFTWords_s3_s32_b300_w500.hdf5'
    sift_32_s4 = 'type4_SIFTWords_s4_s32_b300_w500.hdf5'
    # sift_32_s1234 = 'type4_SIFTWords_s1234_s32_b300_w500.hdf5'
    # generateWords(SIFTFeaFile_b300_s32, groups_s1, saveFolder + sift_32_s1, 500, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s32, groups_s2, saveFolder + sift_32_s2, 500, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s32, groups_s3, saveFolder + sift_32_s3, 500, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s32, groups_s4, saveFolder + sift_32_s4, 500, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s32, groups, saveFolder + sift_32_s1234, wordsNum, feaDim=128)

    sift_48_s1 = 'type4_SIFTWords_s1_s48_b300_w500.hdf5'
    sift_48_s2 = 'type4_SIFTWords_s2_s48_b300_w500.hdf5'
    sift_48_s3 = 'type4_SIFTWords_s3_s48_b300_w500.hdf5'
    sift_48_s4 = 'type4_SIFTWords_s4_s48_b300_w500.hdf5'
    # sift_48_s1234 = 'type4_SIFTWords_s1234_s48_b300_w500.hdf5'
    # generateWords(SIFTFeaFile_b300_s48, groups_s1, saveFolder + sift_48_s1, 500, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s48, groups_s2, saveFolder + sift_48_s2, 500, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s48, groups_s3, saveFolder + sift_48_s3, 500, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s48, groups_s4, saveFolder + sift_48_s4, 500, feaDim=128)
    # generateWords(SIFTFeaFile_b300_s48, groups, saveFolder + sift_48_s1234, 500, feaDim=128)

    # SIFTFeaFile_b300 = '../../Data/Features/type4_SIFTFeatures_s16_300_300_300_300.hdf5'
    # sift_saveName_s1 = 'type4_SIFTWords_s1_s16_300_300_300_300.hdf5'
    # sift_saveName_s2 = 'type4_SIFTWords_s2_s16_300_300_300_300.hdf5'
    # sift_saveName_s3 = 'type4_SIFTWords_s3_s16_300_300_300_300.hdf5'
    # sift_saveName_s4 = 'type4_SIFTWords_s4_s16_300_300_300_300.hdf5'
    # sift_saveName_s1234 = 'type4_SIFTWords_s1234_s16_300_300_300_300.hdf5'
    # generateWords(SIFTFeaFile_b300, groups_s1, saveFolder + sift_saveName_s1, wordsNum, feaDim=128)
    # generateWords(SIFTFeaFile_b300, groups_s2, saveFolder + sift_saveName_s2, wordsNum, feaDim=128)
    # generateWords(SIFTFeaFile_b300, groups_s3, saveFolder + sift_saveName_s3, wordsNum, feaDim=128)
    # generateWords(SIFTFeaFile_b300, groups_s4, saveFolder + sift_saveName_s4, wordsNum, feaDim=128)
    # generateWords(SIFTFeaFile_b300, groups, saveFolder + sift_saveName_s1234, wordsNum, feaDim=128)

    # SDAEFeaFile_same_special_b500 = '../../Data/Features/type4_SDAEFeas_same_mean_s28_b500_special.hdf5'
    # sdae_saveName_s1 = 'type4_SDAEWords_s1_s28_b500_special.hdf5'
    # sdae_saveName_s2 = 'type4_SDAEWords_s2_s28_b500_special.hdf5'
    # sdae_saveName_s3 = 'type4_SDAEWords_s3_s28_b500_special.hdf5'
    # sdae_saveName_s4 = 'type4_SDAEWords_s4_s28_b500_special.hdf5'
    # sdae_saveName_s1234 = 'type4_SDAEWords_s1234_s28_b500_special.hdf5'
    # SDAEFeaFile_same_special_b500 = '../../Data/Features/type4_SDAEFeas_same_mean_s28_b500_special_classification.hdf5'
    # sdae_saveName_s1 = 'type4_SDAEWords_s1_s28_b500_special_classification.hdf5'
    # sdae_saveName_s2 = 'type4_SDAEWords_s2_s28_b500_special_classification.hdf5'
    # sdae_saveName_s3 = 'type4_SDAEWords_s3_s28_b500_special_classification.hdf5'
    # sdae_saveName_s4 = 'type4_SDAEWords_s4_s28_b500_special_classification.hdf5'
    # sdae_saveName_s1234 = 'type4_SDAEWords_s1234_s28_b500_special_classification.hdf5'
    # generateWords(SDAEFeaFile_same_special_b500, groups_s1, saveFolder + sdae_saveName_s1, wordsNum, feaDim=64)
    # generateWords(SDAEFeaFile_same_special_b500, groups_s2, saveFolder + sdae_saveName_s2, wordsNum, feaDim=64)
    # generateWords(SDAEFeaFile_same_special_b500, groups_s3, saveFolder + sdae_saveName_s3, wordsNum, feaDim=64)
    # generateWords(SDAEFeaFile_same_special_b500, groups_s4, saveFolder + sdae_saveName_s4, wordsNum, feaDim=64)
    # generateWords(SDAEFeaFile_same_special_b500, groups, saveFolder + sdae_saveName_s1234, wordsNum, feaDim=64)

    # sift_saveName_h1 = 'type4_SIFTWords_h1.hdf5'
    # sift_saveName_h2 = 'type4_SIFTWords_h2.hdf5'
    # sdae_saveName_h1 = 'type4_SDAEWords_h1.hdf5'
    # sdae_saveName_h2 = 'type4_SDAEWords_h2.hdf5'
    # lbp_saveName_h1 = 'type4_LBPWords_h1.hdf5'
    # lbp_saveName_h2 = 'type4_LBPWords_h2.hdf5'

    # sift_saveName_h1_reduce = 'type4_SIFTWords_h1_reduce.hdf5'
    # sift_saveName_h2_reduce = 'type4_SIFTWords_h2_reduce.hdf5'
    # sdae_saveName_h1_reduce = 'type4_SDAEWords_h1_reduce_sameRatio.hdf5'
    # sdae_saveName_h2_reduce = 'type4_SDAEWords_h2_reduce_sameRatio.hdf5'
    # lbp_saveName_h1_reduce = 'type4_LBPWords_h1_reduce_sameRatio.hdf5'
    # lbp_saveName_h2_reduce = 'type4_LBPWords_h2_reduce_sameRatio.hdf5'
    # sdae_saveName_h1_diff_mean = 'type4_SDAEWords_h1_diff_mean.hdf5'
    # sdae_saveName_h2_diff_mean = 'type4_SDAEWords_h2_diff_mean.hdf5'

    # sift_saveName_h1 = 'type4_SIFTWords_h1_s16_600_300_300_300.hdf5'
    # sift_saveName_h2 = 'type4_SIFTWords_h2_s16_600_300_300_300.hdf5'
    # sdae_saveName_h1 = 'type4_SDAEWords_h1_diff_mean_s16_600_300_300_300.hdf5'
    # sdae_saveName_h2 = 'type4_SDAEWords_h2_diff_mean_s16_600_300_300_300.hdf5'
    # sdae_saveName_h1_s = 'type4_SDAEWords_h1_same_mean_s16_600_300_300_300.hdf5'
    # sdae_saveName_h2_s = 'type4_SDAEWords_h2_same_mean_s16_600_300_300_300.hdf5'
    # lbp_saveName_h1 = 'type4_LBPWords_h1_s16_600_300_300_300.hdf5'
    # lbp_saveName_h2 = 'type4_LBPWords_h2_s16_600_300_300_300.hdf5'

    # sift_saveName_all = 'type4_SIFTWords_all.hdf5'
    # lbp_saveName_all = 'type4_LBPWords_all.hdf5'
    # sdae_saveName_all = 'type4_SDAEWords_all.hdf5'
    # generateWords(siftFeaFile, groups_h1, saveFolder + sift_saveName_h1, wordsNum, feaDim=128)
    # generateWords(siftFeaFile, groups_h2, saveFolder + sift_saveName_h2, wordsNum, feaDim=128)
    # generateWords(LBPFeaFile, groups_h1, saveFolder + lbp _saveName_h1, wordsNum, feaDim=54)
    # generateWords(LBPFeaFile, groups_h2, saveFolder + lbp_saveName_h2, wordsNum, feaDim=54)
    # generateWords(SDAEFeaFile, groups_h1, saveFolder + sdae_saveName_h1, wordsNum, feaDim=128)
    # generateWords(SDAEFeaFile, groups_h2, saveFolder + sdae_saveName_h2, wordsNum, feaDim=128)
    # generateWords(siftFeaFile, groups_all, saveFolder + sift_saveName_all, wordsNum_all, feaDim=128)
    # generateWords(LBPFeaFile, groups_all, saveFolder + lbp_saveName_all, wordsNum_all, feaDim=54)
    # generateWords(SDAEFeaFile, groups_all, saveFolder + sdae_saveName_all, wordsNum_all, feaDim=128)

    # generateWords(siftFeaFile_reduce, groups_h1, saveFolder + sift_saveName_h1_reduce, wordsNum, feaDim=64)
    # generateWords(siftFeaFile_reduce, groups_h2, saveFolder + sift_saveName_h2_reduce, wordsNum, feaDim=64)
    # generateWords(LBPFeaFile_reduce, groups_h1, saveFolder + lbp_saveName_h1_reduce, wordsNum, feaDim=8)
    # generateWords(LBPFeaFile_reduce, groups_h2, saveFolder + lbp_saveName_h2_reduce, wordsNum, feaDim=8)
    # generateWords(SDAEFeaFile_reduce, groups_h1, saveFolder + sdae_saveName_h1_reduce, wordsNum, feaDim=9)
    # generateWords(SDAEFeaFile_reduce, groups_h2, saveFolder + sdae_saveName_h2_reduce, wordsNum, feaDim=9)
    # generateWords(SDAEFeaFile_diff_mean, groups_h1, saveFolder + sdae_saveName_h1_diff_mean, wordsNum, feaDim=128)
    # generateWords(SDAEFeaFile_diff_mean, groups_h2, saveFolder + sdae_saveName_h2_diff_mean, wordsNum, feaDim=128)

    # generateWords(SIFTFeaFile, groups_h1, saveFolder + sift_saveName_h1, wordsNum, feaDim=128)
    # generateWords(SIFTFeaFile, groups_h2, saveFolder + sift_saveName_h2, wordsNum, feaDim=128)
    # generateWords(LBPFeaFile, groups_h1, saveFolder + lbp_saveName_h1, wordsNum, feaDim=54)
    # generateWords(LBPFeaFile, groups_h2, saveFolder + lbp_saveName_h2, wordsNum, feaDim=54)
    # generateWords(SDAEFeaFile, groups_h1, saveFolder + sdae_saveName_h1, wordsNum, feaDim=64)
    # generateWords(SDAEFeaFile, groups_h2, saveFolder + sdae_saveName_h2, wordsNum, feaDim=64)
    # generateWords(SDAEFeaFile_s, groups_h1, saveFolder + sdae_saveName_h1_s, wordsNum, feaDim=64)
    # generateWords(SDAEFeaFile_s, groups_h2, saveFolder + sdae_saveName_h2_s, wordsNum, feaDim=64)

    # cascadeFeas = '../../Data/Features/type4_cascadeFeatures4_s16_600_300_300_300.hdf5'
    # cascade_saveName = '../../Data/Features/type4_cascadeWords_fea4_s16_600_300_300_300.hdf5'
    #
    # groups = ['1', '2', '3', '4']
    # generateWords(cascadeFeas, groups, saveFolder + cascade_saveName, wordsNum)