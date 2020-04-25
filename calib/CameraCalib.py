import numpy as np
import cv2
import glob


#SATRANCLI FOTO ÇEK AYNI KLASORE KOY 10 iyi CEMALLLLLLL ASLFBASLJFÇMBSHICVBIHADCBFk

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


chessX = 3

chessY = 7

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessX*chessY,3), np.float32)
objp[:,:2] = np.mgrid[0:chessY,0:chessX].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

foto = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chessY,chessX),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
    
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (chessY,chessX), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        print(corners)
        print(foto)
        foto += 1

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("ret" + str(ret))
print("\nmtx" + str(mtx))
print("\ndist" + str(dist))
print("\nrvecs" + str(rvecs))
print("\ntvecs" + str(tvecs))
 

cv2.destroyAllWindows()

img = cv2.imread('a.jpg')
h,  w = img.shape[:2]

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

np.savez('ilkCalibre.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

