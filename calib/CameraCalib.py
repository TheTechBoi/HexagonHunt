import numpy as np
import cv2
import glob


#SATRANCLI FOTO ÇEK AYNI KLASORE KOY 10 iyi CEMALLLLLLL ASLFBASLJFÇMBSHICVBIHADCBFk

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


chessX = 7

chessY = 3

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessX*chessY,3), np.float32)

objp[0] = [0, 0, 0]
objp[1] = [3, 0, 0]
objp[2] = [6, 0, 0]
objp[3] = [9, 0, 0]
objp[4] = [12, 0, 0]
objp[5] = [15, 0, 0]
objp[6] = [18, 0, 0]
objp[7] = [0, 3, 0]
objp[8] = [3, 3, 0]
objp[9] = [6, 3, 0]
objp[10] = [9, 3, 0]
objp[11] = [12, 3, 0]
objp[12] = [15, 3, 0]
objp[13] = [18, 3, 0]
objp[14] = [0, 6, 0]
objp[15] = [3, 6, 0]
objp[16] = [6, 6, 0]
objp[17] = [9, 6, 0]
objp[18] = [12, 6, 0]
objp[19] = [15, 6, 0]
objp[20] = [18, 6, 0]

        

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

