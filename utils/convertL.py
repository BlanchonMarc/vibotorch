from os import listdir
import glob
import sys
from PIL import Image
import numpy as np

path = "/Users/marc/Desktop/rbg/image/output/gt"

listfiles = listdir(path)


for ind, infile in enumerate(listfiles):
    try:
        img = Image.open(path + "/" + infile)
        img = img.convert('L')
        img.save(path + "/" + infile)
    except Exception:
        pass
