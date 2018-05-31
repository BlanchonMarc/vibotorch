import Augmentor
from Augmentor import Pipeline

p = Augmentor.Pipeline("/Users/marc/Desktop/rbg/image") # ensure you press enter after this, don't just c&p this code.
p.ground_truth("/Users/marc/Desktop/rbg/gt")
Pipeline.set_seed(100)
p.flip_left_right(probability=0.4)
p.flip_top_bottom(probability=0.4)
p.rotate(probability=0.4, max_left_rotation=25, max_right_rotation=25)
p.random_distortion(probability=0.5, grid_width=6, grid_height=6, magnitude=8)

p.sample(2124)
#
#
# from PIL import Image
# from PIL import ImageFilter
# import os
#
# path, dirs, files = next(os.walk(
#     "/Users/marc/Aquisition Images Final/HSL - Ratcliff1 - Polar"))
# path2, dirs2, files2 = next(os.walk(
#     "/Users/marc/Aquisition Images Final/GroundTruth"))
#
# out_list = []
# in_list = []
# files_list = []
#
# for file in files:
#     if not file.endswith(".png"):
#         files.remove(file)
#
# out_list.append(path + "/output")
# in_list.append(path + "/")
#
# for file in files2:
#     if not file.endswith(".png"):
#         files2.remove(file)
#
# files_list.append(files)
# out_list.append(path2 + "/output")
# in_list.append(path2 + "/")
#
#
# print(out_list)
#
#
# files_list[0] = sorted(files_list[0])
# print(files_list)
# exit()
#
# print(out_list)
#
# for n in range(len(files)):
#     for rot in range(0, 360, 90):
#         for blur in range(0, 6, 2):
#             for files_ in files_list:
#                 for m in range(2):
#                     img = Image.open(in_list[m] + files_[n])
#                     blurred_image = img.filter(
#                         ImageFilter.GaussianBlur(radius=blur))
#                     rotated = blurred_image.rotate(rot)
#                     filenew = files_[n].replace('.png', '')
#                     rotated.save(out_list[m] + "/" + filenew + "-" + str(blur) + "-" + str(rot) + ".png")
