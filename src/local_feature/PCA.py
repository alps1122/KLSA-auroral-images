import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D

def calPcaDimReduceU(featureH5File, groups, feaDim=128):
    f = h5py.File(featureH5File, 'a')
    if 'pca/components' in f:
        print 'pca exist!'
    else:
        feaSet = f.get('feaSet')
        if 'pca' in f:
            pca_group = f.get('pca')
        else:
            pca_group = f.create_group('pca')

        for i in range(len(groups)):
            c = groups[i]
            if isinstance(c, str):
                feas = None
                feas = feaSet.get(c)
                feas = np.array(feas, dtype='float64')
                print 'calculating pca of class ' + c + ' with shape ' + str(feas.shape)

            if isinstance(c, list):
                feas = np.empty((0, feaDim), dtype='float64')
                for cs in c:
                    feat = feaSet.get(cs)
                    feat = np.array(feat, dtype='float64')
                    feas = np.append(feas, feat, axis=0)
                    print 'calculating of class ' + cs + ' with shape ' + str(feas.shape)

            pca = None
            pca = PCA()
            pca.fit(feas)
            # if ratio_num > 1:
            #     keep_components = ratio_num
            # else:
            #     ratios_acc = np.array([var_ratios[:x] for x in range(1, len(var_ratios)+1)])
            #     keep_components = np.argwhere(ratios_acc >= ratio_num) + 1

            # u = pca.components_[:keep_components, :]
            ratios = pca.explained_variance_ratio_
            u = pca.components_
            pca_group.create_dataset('ratios', ratios.shape, 'f', ratios)
            pca_group.create_dataset('components', u.shape, 'f', u)
        f.close()
        print 'pca U saved'
    return 0

def reduceDimension(feaFile, dim_ratio, saveFile):
    f = h5py.File(feaFile, 'r')
    if 'pca/components' not in f:
        print 'pca not calculated, please calculate pac first!'
    else:
        f_r = h5py.File(saveFile, 'w')
        feaSet = f.get('feaSet')
        feaSet_reduce = f_r.create_group('feaSet')
        u = f.get('pca/components')
        if dim_ratio > 1:
            keep_components = dim_ratio
        else:
            ratios = f.get('pca/ratios')
            ratios_acc = np.array([ratios[:x].sum() for x in range(1, len(ratios) + 1)])
            keep_components = np.argwhere(ratios_acc >= dim_ratio).min() + 1

        for c in feaSet:
            feas = feaSet.get(c)
            feas = np.array(feas, dtype='float64')
            u_reduce = u[:keep_components, :]
            feas_reduce = feas.dot(u_reduce.T)

            feaSet_reduce.create_dataset(c, feas_reduce.shape, 'f', feas_reduce)
        f_r.close()
        f.close()
    return 0

def reduceVecFeasDim(feaFile, feaVec, dim_ratio):
    f = h5py.File(feaFile, 'r')
    if 'pca/components' not in f:
        print 'pca not calculated, please calculate pac first!'
    else:
        u = f.get('pca/components')
        if dim_ratio > 1:
            keep_components = dim_ratio
        else:
            ratios = f.get('pca/ratios')
            ratios_acc = np.array([ratios[:x].sum() for x in range(1, len(ratios) + 1)])
            keep_components = np.argwhere(ratios_acc >= dim_ratio).min() + 1

        u_reduce = u[:keep_components, :]
        feas_reduce = feaVec.dot(u_reduce.T)
    return feas_reduce

if __name__ == '__main__':
    SIFTFeaFile = '../../Data/Features/type4_SIFTFeatures.hdf5'
    SDAEFeaFile = '../../Data/Features/type4_SDAEFeas.hdf5'
    LBPFeaFile = '../../Data/Features/type4_LBPFeatures.hdf5'

    groups_all = [['1', '2', '3', '4']]
    calPcaDimReduceU(LBPFeaFile, groups_all, feaDim=54)
    calPcaDimReduceU(SIFTFeaFile, groups_all, feaDim=128)
    calPcaDimReduceU(SDAEFeaFile, groups_all, feaDim=128)

    SIFTFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_reduce.hdf5'
    SDAEFeaFile_reduce = '../../Data/Features/type4_SDAEFeas_reduce_sameRatio.hdf5'
    LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_reduce_sameRatio.hdf5'

    SIFTReduceDim = 64
    f_sift = h5py.File(SIFTFeaFile, 'r')
    ratios_sift = np.array(f_sift.get('pca/ratios'))
    ratios_sift_acc = np.array([ratios_sift[:x].sum() for x in range(1, len(ratios_sift)+1)])
    same_ratio = ratios_sift_acc[SIFTReduceDim-1]
    # reduceDimension(SIFTFeaFile, SIFTReduceDim, SIFTFeaFile_reduce)
    reduceDimension(LBPFeaFile, same_ratio, LBPFeaFile_reduce)
    reduceDimension(SDAEFeaFile, same_ratio, SDAEFeaFile_reduce)
