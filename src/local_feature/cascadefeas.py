import numpy as np
import h5py

def normalize_feas(feaArr):
    feaslen = np.sqrt(np.sum(feaArr ** 2, axis=1))
    feaArr_n = feaArr / feaslen.reshape((feaslen.size, 1))
    return feaArr_n

if __name__ == '__main__':
    siftFeaFile_reduce = '../../Data/Features/type4_SIFTFeatures_reduce.hdf5'
    SDAEFeaFile_reduce = '../../Data/Features/type4_SDAEFeas_reduce_sameRatio.hdf5'
    LBPFeaFile_reduce = '../../Data/Features/type4_LBPFeatures_reduce_sameRatio.hdf5'
    cascade_feas_save_file = '../../Data/Features/type4_cascadeFeatures.hdf5'

    sift_f = h5py.File(siftFeaFile_reduce, 'r')
    sdae_f = h5py.File(SDAEFeaFile_reduce, 'r')
    lbp_f = h5py.File(LBPFeaFile_reduce, 'r')
    for i in sift_f.get('feaSet'):
        print i
    for i in sdae_f.get('feaSet'):
        print i
    for i in lbp_f.get('feaSet'):
        print i

