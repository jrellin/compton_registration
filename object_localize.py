import math
import numpy as np
import scipy.optimize
import cv2


# Helper Methods
def boxcorners(box_size):
    # box_size = [1.5, 1.5, 3] inches
    # Define box corners in box coordinate system.
    half_size_x = box_size[0] / 2.0
    half_size_y = box_size[1] / 2.0
    half_size_z = box_size[2] / 2.0
    corners = np.array(
        [[-half_size_x, -half_size_y, -half_size_z],
         [+half_size_x, -half_size_y, -half_size_z],
         [-half_size_x, +half_size_y, -half_size_z],
         [+half_size_x, +half_size_y, -half_size_z],
         [-half_size_x, -half_size_y, +half_size_z],
         [+half_size_x, -half_size_y, +half_size_z],
         [-half_size_x, +half_size_y, +half_size_z],
         [+half_size_x, +half_size_y, +half_size_z]],
        'float64')
    return corners


def drawBoxOnImage(rotation_vector, translation_vector, camera_matrix, dist_coeffs, image, box_size=(1.5, 1.5, 3)):
    # Draw the box on a given (color) image, given the rotation and
    # translation vector.

    corners = boxcorners(box_size)
    # Project box corners to image plane.
    pts = cv2.projectPoints(
        corners, rotation_vector, translation_vector,
        camera_matrix, dist_coeffs)[0]
    # Draw box on image
    projected_image = image.copy()
    cv2.polylines(
        projected_image,
        np.array([[pts[1][0], pts[0][0], pts[2][0], pts[3][0]],
                  [pts[0][0], pts[1][0], pts[5][0], pts[4][0]],
                  [pts[1][0], pts[3][0], pts[7][0], pts[5][0]],
                  [pts[3][0], pts[2][0], pts[6][0], pts[7][0]],
                  [pts[2][0], pts[0][0], pts[4][0], pts[6][0]],
                  [pts[4][0], pts[5][0], pts[7][0], pts[6][0]]], 'int32'),
        True, (0, 255, 0), 3)
    return projected_image


def projectBox(rotation_vector, translation_vector, camera_matrix, dist_coeffs, image, box_size=(1.5, 1.5, 3)):
    # Project the box to create a mask, given the rotation and translation
    # vector. This function is used in the optimisation loop to compare the
    # projection using the rotation and translation vectors to the original
    # image.
    # box_size = [1.5, 1.5, 3]
    corners = boxcorners(box_size)

    pts = cv2.projectPoints(
        corners, rotation_vector, translation_vector,
        camera_matrix, dist_coeffs)[0]
    # Draw box on image
    projected_image = np.zeros((image.shape[0], image.shape[1], 1),
                               np.uint8)
    cv2.fillConvexPoly(
        projected_image,
        np.array([pts[1][0], pts[0][0], pts[2][0], pts[3][0]], 'int32'),
        (255))
    cv2.fillConvexPoly(
        projected_image,
        np.array([pts[0][0], pts[1][0], pts[5][0], pts[4][0]], 'int32'),
        (255))
    cv2.fillConvexPoly(
        projected_image,
        np.array([pts[1][0], pts[3][0], pts[7][0], pts[5][0]], 'int32'),
        (255))
    cv2.fillConvexPoly(
        projected_image,
        np.array([pts[3][0], pts[2][0], pts[6][0], pts[7][0]], 'int32'),
        (255))
    cv2.fillConvexPoly(
        projected_image,
        np.array([pts[2][0], pts[0][0], pts[4][0], pts[6][0]], 'int32'),
        (255))
    cv2.fillConvexPoly(
        projected_image,
        np.array([pts[4][0], pts[5][0], pts[7][0], pts[6][0]], 'int32'),
        (255))
    # Return projected image.
    return projected_image


def objectiveFunction(x, targetmask, camera_matrix, dist_coeffs):
    # The objective function for the optimisation. Split the x input vector
    # in a rotation and a translation vector, project the box and measure
    # the difference between the projection and the given mask (COG
    # distance, total pixel count (surface) and non-overlapping pixel count
    # (shape difference).

    image_pixels = float(cv2.countNonZero(targetmask))

    # Get rotation and translation vectors and project perfect box.
    rotation_vector = np.array([x[0], x[1], x[2]], 'float64')
    translation_vector = np.array([x[3], x[4], x[5]], 'float64')
    projected = projectBox(rotation_vector, translation_vector, camera_matrix, dist_coeffs,
                           np.zeros(targetmask.shape, np.uint8))

    # Calculate size difference (pixel count).
    projected_pixels = float(cv2.countNonZero(projected))
    pixel_count_difference = \
        ((projected_pixels - image_pixels) / image_pixels) ** 2

    # Calculate overlap difference (pixel count).
    non_overlap = cv2.bitwise_xor(targetmask, projected)
    non_overlap_pixels = float(cv2.countNonZero(non_overlap))
    overlap_difference = non_overlap_pixels / image_pixels

    # Return penalty.
    return pixel_count_difference + overlap_difference


# Main Method
def find_rot_and_trans(object_pic_filename):
    # Load captured image.
    image_bgr = cv2.imread(object_pic_filename, cv2.IMREAD_COLOR)

    # cv2.imshow('image_bgr', image_bgr) # test
    # cv2.waitKey(20000)
    # Convert to HSV color space.
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    YELLOW = {"lower": (25, 100, 50), "upper": (35, 255, 255)}

    # for https://pinetools.com/image-color-picker: ranges are H [0, 360], S [0, 100], V [0, 100]
    # For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
    # to convert from website to opencv: H/2, S * 2.55, V * 2.55
    mask = cv2.inRange(image_hsv, YELLOW["lower"], YELLOW["upper"])

    # cv2.imshow('mask', mask) # test
    # cv2.waitKey(20000)
    print("Number of non-zeros: ", np.count_nonzero(mask))

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16)))
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # This had to be fixed
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(image_bgr, contours, 0, (255, 255, 255), 3)
    targetmask = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 1), np.uint8)
    cv2.drawContours(targetmask, contours, 0, (255), -1)

    # cv2.imshow('target_mask', targetmask)
    # cv2.waitKey(20000)

    # 1.5 inches x 1.5 inches x 3 inches


def test_initial_guess(img_fname, cm, dist_coeffs,
                       box_size=(1.5, 1.5, 3),
                       r0=np.array([0, np.pi/2, 0]),
                       t0=np.array([0, 0, 10.])):
    image = cv2.imread(img_fname, cv2.IMREAD_COLOR)
    proj_img = drawBoxOnImage(r0, t0, cm, dist_coeffs, image, box_size=box_size)
    cv2.imshow('Projection', proj_img)  # test
    cv2.waitKey(20000)

    # drawBoxOnImage(rotation_vector, translation_vector, camera_matrix, dist_coeffs, image, box_size=(1.5, 1.5, 3))


def main():
    import os
    from pathlib import Path
    block_img = 'block_yellow0'
    img_fname = os.path.join(os.getcwd(), 'block_image', block_img + '.jpg')

    camera_matrix_file = 'camera_matrixv0.npz'
    vals = np.load(camera_matrix_file)
    # ret, matrix, distortion, r_vecs, t_vecs
    cm = vals['matrix']
    dist_coeffs = vals['distortion']

    # find_rot_and_trans(img_fname)
    test_initial_guess(img_fname, cm, dist_coeffs)


if __name__ == '__main__':
    main()
