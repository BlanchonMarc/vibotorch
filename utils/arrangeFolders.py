import random
import os
from PIL import Image

random.seed(0)


number_list = list(range(2124))
random.shuffle(number_list)
number_list[:2124]



list1 = number_list[:int(2124 / 3)]
list2 = number_list[int(2124 / 3): int(2 * (2124 / 3))]
list3 = number_list[int(2 * (2124 / 3)):]

path, dirs, files = next(os.walk(
    "/Users/marc/Aquisition Images Final/HSL - Ratcliff1 - Polar/output"))
path2, dirs2, files2 = next(os.walk(
    "/Users/marc/Aquisition Images Final/GroundTruth/output"))

out_listNormal = []
out_listGT = []
in_list = []
files_list = []

for file in files:
    if not file.endswith(".png"):
        files.remove(file)

out_listNormal.append(path + "/test/")

out_listNormal.append(path + "/train/")

out_listNormal.append(path + "/val/")

in_list.append(path + "/")

for file in files2:
    if not file.endswith(".png"):
        files2.remove(file)


out_listGT.append(path2 + "/test/")

out_listGT.append(path2 + "/train/")

out_listGT.append(path2 + "/val/")

in_list.append(path2 + "/")

files_list.append(files)
files_list[0] = sorted(files_list[0])
files = sorted(files)
for n in range(len(files)):
    if n in list1:
        outpath = out_listNormal[0]
        outpath2 = out_listGT[0]

    elif n in list2:
        outpath = out_listNormal[1]
        outpath2 = out_listGT[1]
    else:
        outpath = out_listNormal[2]
        outpath2 = out_listGT[2]

    img = Image.open(in_list[0] + files[n])
    img.save(outpath + files[n])
    img = Image.open(in_list[1] + files[n])
    img.save(outpath2 + files[n])
