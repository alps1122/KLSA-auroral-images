import datetime

def parseNL(path):
    f = open(path, 'r')
    names = []
    labels = []
    lines = f.readlines()
    for line in lines:
        [name, label] = line.split()
        names.append(name)
        labels.append(label)
    return names, labels

def timeDiff(name1, name2):
    # formate: N20031221G030001
    date1 = datetime.datetime(int(name1[1:5]), int(name1[5:7]), int(name1[7:9]), int(name1[10:12]), int(name1[12:14]), int(name1[14:16]))
    date2 = datetime.datetime(int(name2[1:5]), int(name2[5:7]), int(name2[7:9]), int(name2[10:12]), int(name2[12:14]), int(name2[14:16]))
    return (date2-date1).seconds

def sampleImages(names, mode='Uniform', timediff = 60, sampleNum = 500):
    if mode == 'Uniform':
        lastImg = names[0]
        sampledImgs = []
        ids = []
        # sampledLabels = []
        sampledImgs.append(lastImg)
        ids.append(0)
        # sampledLabels.append(labels[0])
        id = 0
        for name in names:
            if timeDiff(sampledImgs[-1], name) >= timediff:
                sampledImgs.append(name)
                ids.append(id)
                # sampledLabels.append(labels[id])
            id = id + 1
        return ids, sampledImgs
    if mode == 'random':
        import random
        # sampledImgs = []
        # ids = []
        # idx = range(len(names))
        random.shuffle(names)
        # for i in range(sampleNum):
        #     sampledImgs.append(names[idx[i]])
            # ids.append(idx[i])
        return names[:sampleNum]


def arrangeToClasses(names, labels, classNum):
    arrangeImgs = {}
    for i in range(classNum):
        arrangeImgs[i+1] = []
    for i in range(len(names)):
        arrangeImgs[int(labels[i])].append(names[i])
    return arrangeImgs

def balanceSample(arrangedImgs, sampleNum):
    for i in range(len(arrangedImgs)):
        arrangedImgs[i+1] = sampleImages(arrangedImgs[i+1], mode='random', sampleNum=sampleNum)
    return arrangedImgs

def compareLabeledFile(file_std, file_compare):
    [names_std, labels_std] = parseNL(file_std)
    [names_compare, labels_compare] = parseNL(file_compare)
    flag = True
    for i in range(len(names_compare)):
        std_idx = names_std.index(names_compare[i])
        if labels_std[std_idx] != labels_compare[i]:
            flag = False
            break
    return  flag