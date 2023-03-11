import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2

sample_image = cv2.imread('./data/shirt.png')
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

low = np.array([30, 90, 30])
high = np.array([90, 240, 240])

mask = cv2.inRange(img, low, high)
contours, hierarchies = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
blank = np.zeros(mask.shape[:2], dtype='uint8')
cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)

if len(contours) != 0:
    # draw in blue the contours that were founded
    cv2.drawContours(img, contours, -1, 255, 3)

    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cx = x + int(w/2)
    cy = y + int(h/2)

    # draw the biggest contour (c) in green
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1)
    cv2.putText(img, "center", (cx - 100, cy - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    print("x: ", cx)
    print("y: ", cy)

result = cv2.bitwise_and(img, img, mask=mask)

plt.axis('off')
plt.imshow(img)
plt.show()
