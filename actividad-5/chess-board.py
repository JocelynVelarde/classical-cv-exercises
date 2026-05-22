import cv2
import numpy as np
import glob
import os

# Our board configuration
CHESSBOARD = (5, 7)       
SQUARE_SIZE = 0.03          

# Tells algorithm to stop when it reached 30 iterations or corner position changes by less than 0.001 pixel
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# We are on a flat plane z = 0, template for the chess board
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = [] 
img_points = [] 

# Images dataset taken from the puzzlebot
images_dir = os.path.join(os.path.dirname(__file__), "Fotos_Chess")
images = glob.glob(os.path.join(images_dir, "*.png"))

gray = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The inner corners this is boolean so true if inner corner detected
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)

    if ret:
        obj_points.append(objp)
        # Corner positions using a 11x11 search window used for calibration
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(refined)

        # Draw the colored lines for visualization purposes
        cv2.drawChessboardCorners(img, CHESSBOARD, refined, ret)
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(500)
        print(f"  [OK] {os.path.basename(fname)}")

cv2.destroyAllWindows()


# Calibrate the camera
# Important to calculate the K which is the 3x3 intristic 
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print(f"RMS reprojection error: {ret:.4f}")
print(f"\nCamera matrix K:\n{K}")
print(f"\nDistortion coefficients (k1, k2, p1, p2, k3):\n{dist.ravel()}")
k1, k2 = dist.ravel()[0], dist.ravel()[1]
print(f"\nk1 = {k1:.6f}")
print(f"k2 = {k2:.6f}")

# Undistort
test_img_path = images[0]  

img = cv2.imread(test_img_path)
h, w = img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, K, dist, None, new_K)

# Crop to valid region
x, y, w2, h2 = roi
if w2 > 0 and h2 > 0:
    undistorted = undistorted[y:y+h2, x:x+w2]

cv2.imshow("Original", img)
cv2.imwrite('original.png', img)
cv2.imshow("Undistorted", undistorted)
cv2.imwrite('undistorted.png', undistorted)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()