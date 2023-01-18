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
def find_rot_and_trans(object_pic_filename,
                       camera_matrix,
                       dist_coeffs,
                       r0=np.array([0, np.pi/2, 0]),
                       tz0=10,
                       camera_fov_degrees=np.array([54.0, 44.0])  # Horizontal x Vertical
                       ):

    if r0.size != 3:
        raise ValueError("{r} does not contain 3 components!".format(r=r0))
    # initial_guess = np.hstack([r0, t0])

    # Load captured image.
    image_bgr = cv2.imread(object_pic_filename, cv2.IMREAD_COLOR)

    # Convert to HSV color space.
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    YELLOW = {"lower": (25, 100, 50), "upper": (35, 255, 255)}

    # for https://pinetools.com/image-color-picker: ranges are H [0, 360], S [0, 100], V [0, 100]
    # For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
    # to convert from website to opencv: H/2, S * 2.55, V * 2.55
    mask = cv2.inRange(image_hsv, YELLOW["lower"], YELLOW["upper"])

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

    # Setting initial guess in x and y based on moments from masked image
    x_fov_dist = tz0 * 2 * np.tan(np.deg2rad(camera_fov_degrees[0]/2.0))
    y_fov_dist = tz0 * 2 * np.tan(np.deg2rad(camera_fov_degrees[1]/2.0))

    Moments = cv2.moments(targetmask)
    x = int(Moments["m10"] / Moments["m00"])
    y = int(Moments["m01"] / Moments["m00"])

    tx0 = ((x/targetmask.shape[1]) - 0.5) * x_fov_dist
    ty0 = ((y/targetmask.shape[0]) - 0.5) * y_fov_dist
    t0 = np.array([tx0, ty0, tz0])

    initial_guess = np.hstack([r0, t0])

    # cv2.imshow('target_mask', targetmask)
    # cv2.waitKey(20000)

    # 1.5 inches x 1.5 inches x 3 inches

    result = scipy.optimize.minimize(objectiveFunction,
                                     initial_guess,
                                     args=(targetmask, camera_matrix, dist_coeffs),
                                     bounds=((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
                                             (None, None), (None, None), (None, None)),
                                     method='Nelder-Mead')
    fit = result.x
    print("Fit: ", fit)
    print("Success: ", result.success)
    print("Termination Message: ", result.message)


def test_initial_guess(img_fname, cm, dist_coeffs,
                       box_size=(1.5, 1.5, 3),
                       r0=np.array([0, np.pi/2, 0]),  # guess 0 -> r0=np.array([0, np.pi/2, 0]), no_rot -> 0, np.pi, 0
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

    # find_rot_and_trans(img_fname, cm, dist_coeffs)  # for finding optimal rotation and translation guesses
    # test_initial_guess(img_fname, cm, dist_coeffs)  # For testing and displaying guesses

    test_initial_guess(img_fname, cm, dist_coeffs,
                       r0=np.array([-3.59663312e-05,  1.58933175e+00, -6.50127592e-05]),
                       t0=np.array([4.15431271e-02, 2.61365716e+00,  9.76298922e+00]))  # if t guess is (0, 3, 10)


if __name__ == '__main__':
    main()
