import numpy as np
import sys
sys.path.insert(0, '../../caffe/python')
import caffe
from caffe import layers as L, params as P
import os
import h5py
from caffe.proto import caffe_pb2
import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2
layerNeuronNum = [28*28, 2000, 1000, 1000, 128]
drop_ratio = 0.2
dataFolder = '../../Data/'
autoencoderSaveFolder = '../../Data/autoEncoder/'
trainData = 'patchDataTrain.txt'
testData = 'PatchDataTest.txt'

def solver(train_net_path, layer_idx, test_net_path=None, base_lr=0.01,
           save_path='../../Data/autoEncoder/auto_encoder_solver.prototxt'):
    s = caffe_pb2.SolverParameter()

    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 100
        s.test_iter.append(128)

    s.type = 'SGD'
    s.base_lr = base_lr
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 5000
    s.momentum = 0.9
    s.weight_decay = 0.0
    s.max_iter = 50000
    s.display = 100
    s.snapshot = 10000
    s.snapshot_prefix = 'save_layer' + str(layer_idx)
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with open(save_path, 'w') as f:
        f.write(str(s))
    return save_path




def layerwise_train_net(h5, batch_size, layer_idx):
    """define layer-wise training net, return the resulting protocol buffer file,
    use str function can turn this result to text file.
    """
    n = caffe.NetSpec()

    n.data, n.label = L.HDF5Data(source=h5, batch_size=batch_size, shuffle=True, ntop=2)
    flatdata = L.Flatten(n.data)
    flatdata_name = 'flatdata'
    n.__setattr__(flatdata_name, flatdata)

    for l in range(layer_idx + 1):
        if l == layer_idx:
            param = learned_param
        else:
            param = frozen_param

        if l == 0:
            if layer_idx == 0:
                drop_noise = L.Dropout(n.flatdata, dropout_param=dict(dropout_ratio=drop_ratio), in_place=False)
                drop_noise_name = 'drop_noise' + str(l + 1)
            else:
                drop_noise = flatdata
                drop_noise_name = flatdata_name

        if l > 0:
            if l == layer_idx:
                drop_noise = L.Dropout(n[relu_en_name], dropout_param=dict(dropout_ratio=drop_ratio), in_place=False)
                drop_noise_name = 'drop_noise' + str(l + 1)
            else:
                drop_noise = relu_en
                drop_noise_name = relu_en_name

        n.__setattr__(drop_noise_name, drop_noise)

        encoder = L.InnerProduct(n[drop_noise_name], num_output=layerNeuronNum[l+1], param=param,
                                 weight_filler=dict(type='gaussian', std=0.005),
                                 bias_filler=dict(type='constant', value=0.1))
        encoder_name = 'encoder' + str(l + 1)
        n.__setattr__(encoder_name, encoder)

        relu_en = L.ReLU(n[encoder_name], in_place=True)
        relu_en_name = 'relu_en' + str(l + 1)
        n.__setattr__(relu_en_name, relu_en)


        if l == layer_idx:
            drop_en = L.Dropout(n[relu_en_name], dropout_param=dict(dropout_ratio=drop_ratio), in_place=True)
            drop_en_name = 'drop_en' + str(l + 1)
            n.__setattr__(drop_en_name, drop_en)

            decoder = L.InnerProduct(n[drop_en_name], num_output=layerNeuronNum[l], param=param,
                                     weight_filler=dict(type='gaussian', std=0.005),
                                     bias_filler=dict(type='constant', value=0.1))
            decoder_name = 'decoder' + str(l + 1)
            n.__setattr__(decoder_name, decoder)

            if l > 0:
                relu_de = L.ReLU(n[decoder_name], dropout_param=dict(dropout_ratio=drop_ratio), in_place=True)
                relu_de_name = 'relu_de' + str(l + 1)
                n.__setattr__(relu_de_name, relu_de)

                n.loss = L.EuclideanLoss(n[relu_de_name], n['relu_en' + str(l)])
            if l == 0:
                n.loss = L.EuclideanLoss(n[decoder_name], n.flatdata)

    return n.to_proto()

def layer_wise_trian():
    layerNum = len(layerNeuronNum) - 1
    for layer_idx in range(layerNum):
        train_net_name = 'autoencoder_train' + str(layer_idx) + '.prototxt'
        test_net_name = 'autoencoder_test' + str(layer_idx) + '.prototxt'
        with open(autoencoderSaveFolder+train_net_name, 'w') as f_tr:
            f_tr.write(str(layerwise_train_net(dataFolder+trainData, 64, layer_idx)))

        with open(autoencoderSaveFolder+test_net_name, 'w') as f_te:
            f_te.write(str(layerwise_train_net(dataFolder+testData, 100, layer_idx)))

    return 0



if __name__ == '__main__':
    patchDataPath = '../../Data/balance500Patch.hdf5'
    # patchData_mean = '../../Data/patchData_mean.txt'
    # fm = open(patchData_mean, 'r')
    # mean_value = float(fm.readline().split(' ')[1])
    f = h5py.File(patchDataPath, 'r')
    data = f.get('data')
    d = np.array(data[0:10, :, :, :])
    # data = f['data']
    print data.shape, d.shape
    for n in f:
        print n

    f.close()

    with open('../../Data/autoEncoder/auto_encoder_train.prototxt', 'w') as f1:
        f1.write(str(layerwise_train_net('/home/ljm/NiuChuang/KLSA-auroral-images/Data/patchDataTrain.txt', 64, 3)))

    # with open('../../Data/autoEncoder/auto_encoder_test.prototxt', 'w') as f1:
    #     f1.write(str(layerwise_train_net('/home/ljm/NiuChuang/KLSA-auroral-images/Data/patchDataTest.txt', 100, 0)))
    #
    # aoto_encoder_solver = solver('../../Data/autoEncoder/auto_encoder_train.prototxt',
    #                              test_net_path='../../DataEncoder/auto_encoder_test.prototxt')





