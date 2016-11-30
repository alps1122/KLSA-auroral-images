import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    lbpFeas = '../../Data/Features/type4_LBPWords_h1.hdf5'
    sdaeFeas = '../../Data/Features/type4_SDAEWords_h1.hdf5'
    siftFeas = '../../Data/Features/type4_SIFTWords_h1.hdf5'

    f = h5py.File(sdaeFeas, 'r')
    w1 = np.array(f.get('1/words'))
    print w1.shape
    print w1.max(), w1.min()
    print np.argwhere(w1 < 1)
    _, axes = plt.subplots(10, 1)
    r = range(500)
    random.shuffle(r)
    for i in range(10):
        axes[i].plot(w1[r[i], :])

    plt.show()