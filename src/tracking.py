import cvzone
from cvzone.ColorModule import ColorFinder
import cv2
import socket
import numpy as np

myColorFinder = ColorFinder(False)
# hsvVals = {'hmin': 33, 'smin': 72, 'vmin': 126, 'hmax': 58, 'smax': 255, 'vmax': 255} #iPad circle
# hsvVals = {'hmin': 15, 'smin': 160, 'vmin': 0, 'hmax': 50, 'smax': 255, 'vmax': 255} Tennis ball
hsvVals = {'hmin': 40, 'smin': 72, 'vmin': 0, 'hmax': 60, 'smax': 255, 'vmax': 255} # Printed circle

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

# FOV = np.deg2rad(73.74)
FOV = np.deg2rad(50)
HALF_FOV = FOV / 2

# Capital letters for world coordinates
# Lowercase letters for camera coordinates

Diameter = 8.7e-2 # cm -> m

def track_shirt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    low = np.array([0, 80, 20])
    high = np.array([50, 255, 120])

    mask = cv2.inRange(img, low, high)
    # cv2.imshow("Mask", mask)
    contours, hierarchies = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(mask.shape[:2], dtype='uint8')
    # cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)

    cx, cy, w, h = 0, 0, 0, 0
    if len(contours) != 0:
        # cv2.drawContours(img, contours, -1, 255, 3)
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cx = x + int(w/2)
        cy = y + int(h/2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1)
        cv2.putText(img, "center", (cx - 100, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
        # print("x: ", cx)
        # print("y: ", cy)
    
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img, cx, cy, w, h

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    success, img = cap.read()
    h, w, _ = img.shape

    while True:
        success, img = cap.read()
        if not success:
            break
        img = track_shirt(img)[0]
        cv2.imshow("Mask", img)
        cv2.waitKey(1)

