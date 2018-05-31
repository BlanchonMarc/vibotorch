from PIL import Image
import os
import sys

path = "/Users/marc/Desktop/rbg/image/"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        if os.path.isfile(path + item):
            try:
                im = Image.open(path + item)
                f, e = os.path.splitext(path + item)
                imResize = im.resize((256, 192), Image.ANTIALIAS)
                w, h = imResize.size
                imResize.crop((1, 1, w - 1, h - 1)).save(f + '.png')
            except Exception:
                pass


if __name__ == "__main__":
    resize()
