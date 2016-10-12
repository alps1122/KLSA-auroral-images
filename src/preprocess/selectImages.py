import src.util.paseLabeledFile as plf

filePath = '../../Data/Alllabel2003_38044.txt'
file_selected = '../../Data/one_in_minute.txt'
file_balance = '../../Data/balanceSampleFrom_one_in_minute.txt'

[names, labels] = plf.parseNL(filePath)

arrangedImgs = plf.arrangeToClasses(names, labels, 4)
for i in range(4):
    print len(arrangedImgs[i+1])

print plf.timeDiff(names[0],names[2])
[ids, sampledImgs] = plf.sampleImages(names)


f_w = open(file_selected, 'w+')

for i in range(len(ids)):
    f_w.write(sampledImgs[i] + ' ' + labels[ids[i]] + '\n')
    # f_w.write(' ')
    # f_w.write(sampledLabels[i])
f_w.close()


[names_s, labels_s] = plf.parseNL(file_selected)
arrangedImgs_s = plf.arrangeToClasses(names_s, labels_s, 4)
for i in range(4):
    print len(arrangedImgs_s[i+1])


balanceImgs = plf.balanceSample(arrangedImgs_s, 500)
f_b = open(file_balance, 'w+')
for c in balanceImgs:
    balanceImgs[c].sort()
    for file in balanceImgs[c]:
        f_b.write(file + ' ' + str(c) + '\n')
f_b.close()

print plf.compareLabeledFile(filePath, file_selected)
print plf.compareLabeledFile(filePath, file_balance)