import numpy as np 
import cv2
from crop import image_resize

# Set the path to the images captured by the left and right cameras
pathL = "./data/stereoL/"
pathR = "./data/stereoR/"

print("Extracting image coordinates of respective 3D pattern ....\n")

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

CamL_id = 1 # Camera ID for left camera
CamR_id = 2 # Camera ID for right camera
 
CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

img_ptsL = []
img_ptsR = []
obj_pts = []

chessboard_size = (9,6)

decimator = 0
while True:
    success, imgL = CamL.read()
    success, imgR = CamR.read()
    imgR = image_resize(imgR, width=imgL.shape[1])
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    outputL = imgL
    outputR = imgR

    if decimator % 15 == 0:
        retR, cornersR =  cv2.findChessboardCorners(outputR,chessboard_size,None)
        retL, cornersL = cv2.findChessboardCorners(outputL,chessboard_size,None)

        if retR and retL:
            obj_pts.append(objp)
            cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
            cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
            cv2.drawChessboardCorners(outputR,chessboard_size,cornersR,retR)
            cv2.drawChessboardCorners(outputL,chessboard_size,cornersL,retL)

            img_ptsL.append(cornersL)
            img_ptsR.append(cornersR)

    cv2.imshow("Image", np.hstack([outputL, outputR]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    decimator+=1


print("Calculating left camera parameters ... ")
# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print("Calculating right camera parameters ... ")
# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))


print("Stereo calibration .....")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                          img_ptsL,
                                                          img_ptsR,
                                                          new_mtxL,
                                                          distL,
                                                          new_mtxR,
                                                          distR,
                                                          imgL_gray.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

# Once we know the transformation between the two cameras we can perform stereo rectification
# StereoRectify function
rectify_scale= 1 # if 0 image croped, if 1 image not croped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                 imgL_gray.shape[::-1], Rot, Trns,
                                                 rectify_scale,(0,0))

# Use the rotation matrixes for stereo rectification and camera intrinsics for undistorting the image
# Compute the rectification map (mapping between the original image pixels and 
# their transformed values after applying rectification and undistortion) for left and right camera frames
Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                             imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                              imgR_gray.shape[::-1], cv2.CV_16SC2)


print("Saving paraeters ......")
cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
cv_file.release()