import video_converter
import matplotlib.pyplot as plt
import glob
import utils

from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from camera_calibration import calibrate, undistort_dir

import scale_train
import detection
import numpy as np

color_space = 'YCrCb'

if __name__ == '__main__':
    data_prefix = "/Users/andriicherniak/Desktop/CarND-Vehicle-Detection-master"
    input_video = "{}/project_video.mp4".format(data_prefix)
    output_video = "{}/output_video.mp4".format(data_prefix)

    video_images = "{}/video_images".format(data_prefix)
    calibration_images = "{}/camera_cal".format(data_prefix)
    undistorted_images = "{}/undistorted_images".format(data_prefix)
    heat_images = "{}/heat_images".format(data_prefix)

    test_images = "{}/test_images".format(data_prefix)
    output_images = "{}/output_images".format(data_prefix)

    vehicles_images = "{}/vehicles".format(data_prefix)
    nonvehicles_images = "{}/non-vehicles".format(data_prefix)

    scaler_model = "{}/scaler.pkl".format(data_prefix)
    svc_model = "{}/svc.pkl".format(data_prefix)

    # video_converter.video_to_images(
    #      video_path=input_video,
    #      images_path=video_images)

    # mtx, dist = calibrate(
    #     img_path=calibration_images,
    #     img_prefix="calibration")
    #
    # undistort_dir(
    #     input_path=video_images,
    #     output_path=undistorted_images,
    #     mtx=mtx,
    #     dist=dist)


    # cars = list(glob.glob("{}/*/*.png".format(vehicles_images)))
    # notcars = list(glob.glob("{}/*/*.png".format(nonvehicles_images)))
    # scale_train.train(cars=cars, notcars=notcars, scaler_model=scaler_model, svc_model=svc_model)

    X_scaler = joblib.load(scaler_model)
    svc = joblib.load(svc_model)

    ystart = 400
    ystop = 656
    scale_list = [0.8, 1, 1.5, 2, 2.5, 3]
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    heat_history = 3

    image_list = glob.glob("{}/*.jpg".format(undistorted_images))

    for image_path in image_list[0:0]:
        print(image_path)
        img = utils.read_image(image_path)
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        for scale in scale_list:
            heat = detection.find_cars(
                img=img,
                ystart=ystart,
                ystop=ystop,
                scale=scale,
                svc=svc,
                X_scaler=X_scaler,
                orient=orient,
                pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                heat=heat)

        f_name = utils.file_name(image_path)
        out_path = "{}/{}.pkl".format(heat_images, f_name)
        joblib.dump(heat, out_path)

    heat_list = list(glob.glob("{}/*.pkl".format(heat_images)))
    # for i in range(heat_history - 1, len(heat_list)):
    #     print(image_list[i])
    #     img = utils.read_image(image_list[i])
    #     cumulative_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    #
    #     for j in range(0, heat_history):
    #         # print(heat_list[i-j])
    #         heat = joblib.load(heat_list[i - j])
    #         cumulative_heat += heat
    #
    #     cumulative_heat = detection.apply_threshold(cumulative_heat, 3)
    #     heatmap = np.clip(cumulative_heat, 0, 255)
    #     labels = label(heatmap)
    #     draw_img = detection.draw_labeled_bboxes(np.copy(img), labels)
    #
    #     o_path = "{}/{}".format(output_images, utils.file_name(image_list[i]))
    #     utils.save_image(draw_img, o_path)

    video_converter.images_to_video(
        images_path=output_images,
        video_path=output_video)
