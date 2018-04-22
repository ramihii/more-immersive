import numpy as np
import cv2
#import glob

# marker dictionary
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


# Load previously saved camera calibration data
with np.load('a.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


def readAndDrawMarkers(frame):
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, d)
    if ids is None:
        cv2.imshow("test", frame)
        return cv2.waitKey(25)

    # highlight detected markers and draw ids
    image = cv2.aruco.drawDetectedMarkers(frame, corners, ids, (255, 0, 255) )
    
    # rotation and translation vectors
    # 0.07 is marker size in meters
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers( corners, 0.07, mtx, dist )
    
    # draw axis for the markers
    for i in range( len(ids)):
        image = cv2.aruco.drawAxis(image, mtx, dist, rvecs[i], tvecs[i], 0.05)

    cv2.imshow('test', image)
    return cv2.waitKey(25)

# webcam
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        break
    k = readAndDrawMarkers(frame)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break


cam.release()
cv2.destroyAllWindows()

'''
# reading video file

fname = 'immersive.mp4'
cap = cv2.VideoCapture(fname)

# img = cv2.imread(fname)



while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        readAndDrawMarkers(frame)
    else:
        break

cap.release()
cv2.destroyAllWindows()
'''
