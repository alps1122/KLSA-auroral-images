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
arrangedImgs_s = plf.arrangeToClasses(names_s, labels_s, class_num)

print 'one in minute number:'
for i in range(class_num):
    print 'NO. class ' + str(i+1), len(arrangedImgs_s[str(i+1)])

balanceImgs = plf.balanceSample(arrangedImgs_s, select_num_perclass)
f_b = open(file_balance, 'w+')

print 'balance selected number:'
for c in balanceImgs:
    print 'NO. class ' + str(i + 1), len(arrangedImgs_s[str(i + 1)])
    balanceImgs[c].sort()
    for file in balanceImgs[c]:
        f_b.write(file + ' ' + str(c) + '\n')
f_b.close()

print 'test sampled labeled file:'
print plf.compareLabeledFile(filePath, file_selected)
print plf.compareLabeledFile(filePath, file_balance)