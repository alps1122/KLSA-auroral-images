import numpy as np
import h5py

def calEucDistance(wordsFile):
    fw = h5py.File(wordsFile, 'r')
    # classNum = len(fw)
    # for c in fw:
    w1 = fw.get('/1/words')
    w2 = fw.get('/2/words')
    w1 = np.array(w1)
    w2 = np.array(w2)
    num_words = w1.shape[0]

    distance11 = np.zeros((num_words, num_words))
    distance12 = np.zeros((num_words, num_words))
    distance22 = np.zeros((num_words, num_words))

    for i in range(num_words):
        w1i = w1[i]
        w2i = w2[i]
        for j in range(num_words):
            w1j = w1[j]
            w2j = w2[j]
            distance12[i, j] = np.linalg.norm(w1i - w2j)
            distance11[i, j] = np.linalg.norm(w1i - w1j)
            distance22[i, j] = np.linalg.norm(w2i - w2j)
    savePath = '../../Data/Features/SIFT_Distance.hdf5'
    fd = h5py.File(savePath, 'w')
    fd.create_dataset('distance12', distance12.shape, 'f', distance12)
    fd.create_dataset('distance11', distance11.shape, 'f', distance11)
    fd.create_dataset('distance22', distance22.shape, 'f', distance22)
    print 'distance11:'
    print 'max: ' + str(distance11.max())
    print 'min: ' + str(distance11.min())
    print 'distance12:'
    print 'max: ' + str(distance12.max())
    print 'min: ' + str(distance12.min())
    print 'distance22:'
    print 'max: ' + str(distance22.max())
    print 'min: ' + str(distance22.min())
    fd.close()
    fw.close()
    return distance11, distance12, distance22

def calCommonVector(wordsFile):
    distance11, distance12, distance22 = calEucDistance(wordsFile)
    # dis11_min = distance11.min()
    dis11_max = distance11.max()
    dis22_max = distance22.max()
    dis12_min = distance12.min()
    dis12_max = distance12.max()

    word_num = distance11.shape[0]
    dis11_min = distance11.flatten()
    dis11_min.sort()
    dis11_min = dis11_min[word_num]

    dis22_min = distance22.flatten()
    dis22_min.sort()
    dis22_min = dis22_min[word_num]

    print 'distance11 min: ' + str(dis11_min) + ', max: ' + str(dis11_max)
    print 'distance22 min: ' + str(dis22_min) + ', max: ' + str(dis22_max)
    print 'distance12 min: ' + str(dis12_min) + ', max: ' + str(dis12_max)

    common_threshold = max(dis11_min, dis22_min)
    print 'common threshold: ' + str(common_threshold)

    common_index = distance12 < common_threshold
    common_index = common_index - np.eye(word_num, word_num, dtype='i')

    c1_common = common_index.sum(1)
    c2_common = common_index.sum(0)
    c1_common_v = np.argwhere(c1_common > 0)
    c2_common_v = np.argwhere(c2_common > 0)

    c1_common_num = (c1_common > 0).sum()
    c2_common_num = (c2_common > 0).sum()

    print 'c1_common_num: ' + str(c1_common_num)
    print 'c2_common_num: ' + str(c2_common_num)
    return c1_common_v, c2_common_v

if __name__ == '__main__':
    # wordsFile = '../../Data/Features/SIFTWords_12.hdf5'
    wordsFile = '../../Data/Features/SIFTWords_h1.hdf5'
    distance11, distance12, distance22 = calEucDistance(wordsFile)

    c1_common_v, c2_common_v = calCommonVector(wordsFile)

    print 'c1_common_v: '
    print c1_common_v
    print 'c2_common_v: '
    print c2_common_v