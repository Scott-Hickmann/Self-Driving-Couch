import numpy as np
import cv2

imgL = cv2.imread('content/room_l.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('content/room_r.jpg', cv2.IMREAD_GRAYSCALE)
# imgL = cv2.imread('content/tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
# imgR = cv2.imread('content/tsukuba_r.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Java', imgL)
# cv2.waitKey(0)
cv2.imshow('Java', imgR)
# cv2.waitKey(0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgL, imgR)
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('Java', disparity)
cv2.waitKey(0)