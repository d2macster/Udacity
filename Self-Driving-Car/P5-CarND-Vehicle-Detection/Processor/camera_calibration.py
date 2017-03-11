import numpy as np
import cv2
import glob
import os


def calibrate(img_path, img_prefix):
    nx = 9
    ny = 6

    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob("{}/{}*.jpg".format(img_path, img_prefix))

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    test_img_input = "{}/{}1.jpg".format(img_path, img_prefix)

    img = cv2.imread(test_img_input)
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def undistort_dir(input_path, output_path, mtx, dist):
    images = glob.glob("{}/*.jpg".format(input_path))
    for path in images:
        print(path)
        img_name = os.path.split(path)[1]
        out = "{}/{}".format(output_path, img_name)
        img = cv2.imread(path)
        img = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite(out, img)


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)