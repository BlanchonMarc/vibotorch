import re
import csv


textfile = '/Users/marc/Anakim_Save/Pola1/log.txt'

lines = []
with open(textfile, 'rt') as in_file:
    for line in in_file:
        lines.append(line)
# print(lines)

number = []

for ind in range(len(lines)):
    # print(lines[ind])
    s = [s for s in re.findall(r'\b\d+\b', lines[ind])]
    # print(s)
    if len(s) > 1:
        number.append(float(s[0] + '.' + s[1]))
    elif len(s) == 1:
        number.append(float(s[0]))
    else:
        pass
# print(number)
IoU = []
MAcc = []
OAcc = []
F1 = []

listIoU = [n for n in range(12, len(lines), 12)]
listMAcc = [n for n in range(11, len(lines), 12)]
listOAcc = [n for n in range(10, len(lines), 12)]
listF1 = [n for n in range(8, len(lines), 12)]
for i in range(len(number)):

    if i in listIoU:
        IoU.append(number[i])
    elif i in listMAcc:
        MAcc.append(number[i])
    elif i in listOAcc:
        OAcc.append(number[i])
    elif i in listF1:
        F1.append(number[i])

IOUlog = '/Users/marc/Anakim_Save/Pola1/iou.csv'
MAcclog = '/Users/marc/Anakim_Save/Pola1/macc.csv'
OAcclog = '/Users/marc/Anakim_Save/Pola1/oacc.csv'
F1log = '/Users/marc/Anakim_Save/Pola1/f1.csv'

with open(IOUlog, "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(IoU)
with open(MAcclog, "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(MAcc)
with open(OAcclog, "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(OAcc)
with open(F1log, "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(F1)
