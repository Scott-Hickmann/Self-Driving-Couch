from crop import image_resize
import numpy as np
import cv2

left_cap = cv2.VideoCapture(2)
right_cap = cv2.VideoCapture(3)

while True:
    left_success, left_img = left_cap.read()
    right_success, right_img = right_cap.read()
    if left_success and right_success:
        right_img = image_resize(right_img, width=left_img.shape[1])
        cv2.imshow("Image", np.hstack([left_img, right_img]))
    cv2.waitKey(1)