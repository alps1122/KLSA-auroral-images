'''
Chuang Niu, niuchuang@stu.xidian.edu.cn
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../caffe/python')
import caffe
from caffe import layers as L, params as P
import os
import h5py
from caffe.proto import caffe_pb2
import src.preprocess.esg as esg
import src.util.paseLabeledFile as plf

if __name__ == '__main__':
    model = '../../Data/final_0.01.caffemodel'
