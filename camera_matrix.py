import glob
import cv2
import os
from pathlib import Path
import numpy as np


def generate_camera_matrix(chessboard_patten=(6, 8), pattern_folder="chessboard_images", show=True, save=False):
    """Chessboard_pattern is (vertical, horizontal) number of squares of test pattern.
    Pattern folder is the relative folder of test pattern images.
    Show briefly displays the mapped images. Save saves them if set to True."""

    vpts, hpts = chessboard_patten
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # path to chessboard images
    p = Path(os.getcwd()) / pattern_folder
    print(p)

    # path to processed images
    mapped = p / 'maps'

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((vpts * hpts, 3), np.float32)
    objp[:, :2] = np.mgrid[0:vpts, 0:hpts].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = p.glob('*.jpg')
    success = 0
    failure = 0

    for fname in images:
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale image

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (vpts, hpts), None)

        # If found, add object points, image points (after refining them)
        if ret:
            success += 1
            print(fname.name + ": Found object points!")
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            if not show:
                continue

            cv2.drawChessboardCorners(img, (vpts, hpts), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            if save:
                cv2.imwrite(str(mapped / ('mapped_' + fname.name)), img)
        else:
            failure += 1
            print(fname.name + ": Did not find object points!")

    if success <= 0:
        print("No successes! Nothing to return!")
        return
    print("Total successes: {s}. Total failures: {f}.".format(s=success, f=failure))
    cv2.destroyAllWindows()

    im_shape = img.shape[1::-1]  # gray.shape[::-1]

    # ret, mtx, dist, rvecs, tvecs
    return cv2.calibrateCamera(objpoints, imgpoints, im_shape, None, None)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    save_matrix = False
    save_matrix_ext = 'v0'  # jan 13
    ret, matrix, distortion, r_vecs, t_vecs = generate_camera_matrix(show=False, save=False)

    print("ret:")
    print(ret)

    print(" Camera matrix:")
    print(matrix)

    print("\n Distortion coefficient:")
    print(distortion)

    print("\n Rotation Vectors:")
    print(r_vecs)

    print("\n Translation Vectors:")
    print(t_vecs)

    if save_matrix:
        np.savez('camera_matrix' + save_matrix_ext, ret=ret, matrix=matrix,distortion=distortion, r_vecs=r_vecs,
                 t_vecs=t_vecs)
