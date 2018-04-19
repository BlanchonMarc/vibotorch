import numpy as np
import imutils
import cv2


class Stitcher:

    def __init__(self):
        self.isv3 = imutils.is_cv3()

    def stitch(self, imageA, imageB, ratio=0.75,
               reprojThresh=4.0, showMatches=False):
        sift = cv2.xfeatures2d.SIFT_create()
        (kpsA, featuresA) = sift.detectAndCompute(imageA, None)
        (kpsB, featuresB) = sift.detectAndCompute(imageB, None)
        MIN_MATCH_COUNT = 10
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(featuresA, featuresB, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        src_pts = []
        dst_pts = []
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kpsA[m.queryIdx].pt
                                  for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpsB[m.trainIdx].pt
                                  for m in good]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(src_pts, dst_pts)

        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1],
                                      imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)
        return result


if __name__ == '__main__':
    stitchObj = Stitcher()
    img2 = cv2.imread("/Users/marc/Downloads/testimage/NIR.tiff", 0)
    img1 = cv2.imread("/Users/marc/Downloads/testimage/IDS.tiff", 0)
    result = stitchObj.stitch(imageA=img1, imageB=img2)
    cv2.imshow('result', result)
    cv2.imshow('CACHE', result)
