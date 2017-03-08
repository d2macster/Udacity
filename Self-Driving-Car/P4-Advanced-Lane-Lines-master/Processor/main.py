import video_converter
from camera_calibration import calibrate, undistort_dir
import utils
import lane_operations
import thresholder
import warper
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':
    data_prefix = "/Users/andriicherniak/Desktop/CarND-Advanced-Lane-Lines-master"
    input_video = "{}/project_video.mp4".format(data_prefix)
    output_video = "{}/output_video.mp4".format(data_prefix)

    video_images = "{}/video_images".format(data_prefix)

    calibration_images = "{}/camera_cal".format(data_prefix)
    undistorted_images = "{}/undistorted_images".format(data_prefix)
    test_images = "{}/test_images".format(data_prefix)
    output_images = "{}/output_images".format(data_prefix)

    # video_converter.video_to_images(
    #     video_path=input_video,
    #     images_path=video_images,
    #     image_prefix="img")

    # mtx, dist = calibrate(
    #     img_path=calibration_images,
    #     img_prefix="calibration")
    #
    # undistort_dir(
    #     input_path=video_images,
    #     output_path=undistorted_images,
    #     mtx=mtx,
    #     dist=dist)

    right_fit_prev = None
    left_fit_prev = None

    for img_path in utils.get_image_list(undistorted_images):
        print(img_path)
        img = utils.read_image(img_path[0])
        warped = warper.warp(img)

        hsl_warped = thresholder.get_thresholded_hsl(warped)
        hsv_warped = thresholder.get_thresholded_hsv(warped)

        lanes, left_fit_hsl, right_fit_hsl = lane_operations.detect_lane(hsl_warped)
        lanes, left_fit_hsv, right_fit_hsv = lane_operations.detect_lane(hsv_warped)

        diff_r = 1.0 * abs(right_fit_hsv[0] - right_fit_hsl[0])
        diff_l = 1.0 * abs(left_fit_hsv[0] - left_fit_hsl[0])

        if right_fit_prev is None:
            right_fit_prev = 0.5 * right_fit_hsl + 0.5 * right_fit_hsv

        if left_fit_prev is None:
            left_fit_prev = 0.5 * left_fit_hsl + 0.5 * left_fit_hsv

        if diff_r > 0.0001:
            diff_r_r = 1.0 * abs(right_fit_hsl[0] - right_fit_prev[0])
            if diff_r_r < 0.0005:
                right_fit = 0.2 * right_fit_hsl + 0.8 * right_fit_prev
            else:
                right_fit = right_fit_prev
        else:
            right_fit = 0.5 * right_fit_hsl + 0.5 * right_fit_hsv

        if diff_l > 0.0001:
            diff_l_l = 1.0 * abs(left_fit_hsl[0] - left_fit_prev[0])
            if diff_l_l < 0.0005:
                left_fit = 0.2 * left_fit_hsl + 0.8 * left_fit_prev
            else:
                left_fit = left_fit_prev
        else:
            left_fit = 0.5 * left_fit_hsl + 0.5 * left_fit_hsv

        left_fit_prev = left_fit
        right_fit_prev = right_fit

        left_curve = lane_operations.get_curvature(left_fit, img.shape[0] / 2)
        right_curve = lane_operations.get_curvature(right_fit, img.shape[0] / 2)

        color_warp = np.zeros_like(lanes)

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        lane = warper.unwarp(color_warp)
        result = cv2.addWeighted(img, 1, lane, 0.3, 0)

        ## Compute intercepts
        left_bot = lane_operations.get_val(img.shape[0], left_fit)
        right_bot = lane_operations.get_val(img.shape[0], right_fit)

        val_center = (left_bot + right_bot) / 2.0

        ## highway width = 3.7 meter , ~ 720 pixels on the photo
        # 1 pixel = 370 /720 cm = 0.51
        dist_offset = val_center - img.shape[1] / 2
        dist_offset = np.round(0.51 * dist_offset, 2)

        str_curv = 'Curvature: L = {} km, R = {} km, offset = {} cm'.format(
            np.round(0.001 * left_curve, 1),
            np.round(0.001 * right_curve, 1),
            dist_offset
        )
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(result, str_curv, (30, 60), font, 1, (0, 255, 0), 2)

        output_path = "{}/{}".format(output_images, img_path[1][1])
        utils.save_image(result, output_path)

    video_converter.images_to_video(output_images, output_video)
