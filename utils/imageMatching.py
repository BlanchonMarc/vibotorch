# # import cv2
# # import numpy as np
# #
# # # Read the images to be aligned
# # im1 = cv2.imread("/Users/marc/Downloads/testimage/IDS.tiff")
# # im2 = cv2.imread("/Users/marc/Downloads/testimage/NIR.tiff")
# #
# # # Convert images to grayscale
# # im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
# # im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
# #
# # # Find size of image1
# # sz = im1.shape
# #
# # # Define the motion model
# # warp_mode = cv2.MOTION_TRANSLATION
# #
# # # Define 2x3 or 3x3 matrices and initialize the matrix to identity
# # if warp_mode == cv2.MOTION_HOMOGRAPHY:
# #     warp_matrix = np.eye(3, 3, dtype=np.float32)
# # else:
# #     warp_matrix = np.eye(2, 3, dtype=np.float32)
# #
# # # Specify the number of iterations.
# # number_of_iterations = 500000
# #
# # # Specify the threshold of the increment
# # # in the correlation coefficient between two iterations
# # termination_eps = 1e-10
# #
# # # Define termination criteria
# # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
# #
# # # Run the ECC algorithm. The results are stored in warp_matrix.
# # (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
# #
# # if warp_mode == cv2.MOTION_HOMOGRAPHY:
# #     # Use warpPerspective for Homography
# #     im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
# # else:
# #     # Use warpAffine for Translation, Euclidean and Affine
# #     im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
# #
# # # Show final results
# # cv2.imshow("Image 1", im1)
# # cv2.imshow("Image 2", im2)
# # cv2.imshow("Aligned Image 2", im2_aligned)
# # cv2.waitKey(0)
#
#
# import os
# import cv2
# import numpy as np
# from time import time
#
#
#
# # Align and stack images with ECC method
# # Slower but more accurate
# def stackImagesECC(file_list):
#     M = np.eye(3, 3, dtype=np.float32)
#
#     first_image = None
#     stacked_image = None
#
#     for file in file_list:
#         image = cv2.imread(file,1).astype(np.float32) / 255
#         print(file)
#         if first_image is None:
#             # convert to gray scale floating point image
#             first_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#             stacked_image = image
#         else:
#             # Estimate perspective transform
#             s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)
#             w, h, _ = image.shape
#             # Align image to first image
#             image = cv2.warpPerspective(image, M, (h, w))
#             stacked_image += image
#
#     stacked_image /= len(file_list)
#     stacked_image = (stacked_image*255).astype(np.uint8)
#     return stacked_image
#
#
# # Align and stack images by matching ORB keypoints
# # Faster but less accurate
# def stackImagesKeypointMatching(file_list):
#
#     orb = cv2.ORB_create()
#
#     # disable OpenCL to because of bug in ORB in OpenCV 3.1
#     cv2.ocl.setUseOpenCL(False)
#
#     stacked_image = None
#     first_image = None
#     first_kp = None
#     first_des = None
#     for file in file_list:
#         print(file)
#         image = cv2.imread(file,1)
#         imageF = image.astype(np.float32) / 255
#
#         # compute the descriptors with ORB
#         kp = orb.detect(image, None)
#         kp, des = orb.compute(image, kp)
#
#         # create BFMatcher object
#         matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
#         if first_image is None:
#             # Save keypoints for first image
#             stacked_image = imageF
#             first_image = image
#             first_kp = kp
#             first_des = des
#         else:
#              # Find matches and sort them in the order of their distance
#             matches = matcher.match(first_des, des)
#             matches = sorted(matches, key=lambda x: x.distance)
#
#             src_pts = np.float32(
#                 [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#             dst_pts = np.float32(
#                 [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#
#             # Estimate perspective transformation
#             M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
#             w, h, _ = imageF.shape
#             imageF = cv2.warpPerspective(imageF, M, (h, w))
#             stacked_image += imageF
#
#     stacked_image /= len(file_list)
#     stacked_image = (stacked_image*255).astype(np.uint8)
#     return stacked_image
#
# # ===== MAIN =====
# # Read all files in directory
# import argparse
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('input_dir', help='Input directory of images ()')
#     parser.add_argument('output_image', help='Output image name')
#     parser.add_argument('--method', help='Stacking method ORB (faster) or ECC (more precise)')
#     parser.add_argument('--show', help='Show result image',action='store_true')
#     args = parser.parse_args()
#
#     image_folder = args.input_dir
#     if not os.path.exists(image_folder):
#         print("ERROR {} not found!".format(image_folder))
#         exit()
#
#     file_list = os.listdir(image_folder)
#     file_list = [os.path.join(image_folder, x)
#                  for x in file_list if x.endswith(('.jpg', '.png','.bmp'))]
#
#     if args.method is not None:
#         method = str(args.method)
#     else:
#         method = 'KP'
#
#     tic = time()
#
#     if method == 'ECC':
#         # Stack images using ECC method
#         description = "Stacking images using ECC method"
#         print(description)
#         stacked_image = stackImagesECC(file_list)
#
#     elif method == 'ORB':
#         #Stack images using ORB keypoint method
#         description = "Stacking images using ORB method"
#         print(description)
#         stacked_image = stackImagesKeypointMatching(file_list)
#
#     else:
#         print("ERROR: method {} not found!".format(method))
#         exit()
#
#     print("Stacked {0} in {1} seconds".format(len(file_list), (time()-tic) ))
#
#     print("Saved {}".format(args.output_image))
#     cv2.imwrite(str(args.output_image),stacked_image)
#
#     # Show image
#     if args.show:
#         cv2.imshow(description, stacked_image)
#         cv2.waitKey(0)


import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


# img1 = cv2.imread("/Users/marc/Aquisition Images/Kinect/image_00201.tiff", 0)
# img2 = cv2.imread("/Users/marc/Aquisition Images/NIR/image_00201.tiff", 0)

img1 = cv2.imread("/Users/marc/Downloads/testimage/IDS.tiff", 0)
img2 = cv2.imread("/Users/marc/Downloads/testimage/NIR.tiff", 0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
src_pts = []
dst_pts = []
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

cv2.imshow("Matches", img3)


# Calculate Homography
h, status = cv2.findHomography(dst_pts, src_pts)

print(h)

# cv2.imshow("Cache", img3)

# # Warp source image to destination based on homography
im_out = cv2.warpPerspective(img2, h, (img1.shape[1], img1.shape[0]))

# # Display images
# cv2.imshow("Source Image", img2)
# cv2.imshow("Destination Image", img1)
cv2.imshow("Warped Source Image", im_out)
cv2.imshow("Cache", im_out)
