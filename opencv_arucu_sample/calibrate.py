import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# checkerboard dimensions
cbrow = 9
cbcol = 7

# square size, if you know checkerboard square size, add it in meters.
# if not use 1.0
squareSize = 0.023

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
pattern = (cbrow, cbcol)
objp = np.zeros((np.prod(pattern), 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)
objp *= squareSize

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./calibration-img/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)
    print(fname, ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,9), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(50)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('\ncamera matrix:')
print(mtx)

print('\ndistortion coefficients:')
print(dist)
np.savez('a.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
