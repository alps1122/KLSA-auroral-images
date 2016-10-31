"""According to some self-define rules, this function selects the images for
following experiments."""

import src.util.paseLabeledFile as plf

filePath = '../../Data/Alllabel2003_38044.txt'
file_selected = '../../Data/one_in_minute.txt'
file_balance = '../../Data/balanceSampleFrom_one_in_minute.txt'
class_num = 4
select_num_perclass = 500

[names, labels] = plf.parseNL(filePath)

arrangedImgs = plf.arrangeToClasses(names, labels, class_num)

print 'total labeled number:'
for i in range(4):
    print 'NO. class ' + str(i+1), len(arrangedImgs[str(i+1)])

# print plf.timeDiff(names[0], names[2])
[ids, sampledImgs] = plf.sampleImages(names)

f_w = open(file_selected, 'w+')

for i in range(len(ids)):
    f_w.write(sampledImgs[i] + ' ' + labels[ids[i]] + '\n')

f_w.close()

[names_s, labels_s] = plf.parseNL(file_selected)
arrangedImgs_s = plf.arrangeToClasses(names_s, labels_s, class_num, [['1'], ['2'], ['3'], ['4']])

print 'one in minute number:'
for i in range(class_num):
    print 'NO. class ' + str(i+1), len(arrangedImgs_s[str(i+1)])

balanceImgs = plf.balanceSample(arrangedImgs_s, select_num_perclass)
f_b = open(file_balance, 'w+')

print 'balance selected number:'
for c in balanceImgs:
    print 'NO. class ' + c, len(balanceImgs[c])
    balanceImgs[c].sort()
    for file in balanceImgs[c]:
        f_b.write(file + ' ' + str(c) + '\n')
f_b.close()

print 'test sampled labeled file:'
print plf.compareLabeledFile(filePath, file_selected)
print plf.compareLabeledFile(filePath, file_balance)

type3_file = '../../Data/type3_1000_500_500.txt'
f2 = open(type3_file, 'w')

arrangedImgs_s3, rawTypes = plf.arrangeToClasses(names_s, labels_s, 3, [['1'], ['2'], ['3']])
print 'class1: ' + str(len(arrangedImgs_s3['1']))
print 'class2: ' + str(len(arrangedImgs_s3['2']))
print 'class3: ' + str(len(arrangedImgs_s3['3']))

balance3Imgs = plf.balanceSample(arrangedImgs_s3, 1000)
for c in balance3Imgs:
    print 'NO. class ' + c, len(balance3Imgs[c])

imgs_s23 = balance3Imgs.copy()
imgs_s23.pop('1')
imgs_s23 = plf.balanceSample(imgs_s23, 500)

imgs_s123 = {}
imgs_s123['1'] = balance3Imgs['1']
imgs_s123['2'] = imgs_s23['2']
imgs_s123['3'] = imgs_s23['3']

for c in imgs_s123:
    print 'NO. class ' + c, len(imgs_s123[c])
    imgs_s123[c].sort()
    for file in imgs_s123[c]:
        f2.write(file + ' ' + str(c) + '\n')
f2.close()
print plf.compareLabeledFile(filePath, type3_file)