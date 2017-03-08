import numpy as np
import matplotlib.pyplot as plt
import cv2


def swap(s1, s2):
    return s2, s1


def find_peaks(data):
    result = set()
    dl = list(enumerate(data))
    l = len(data)
    for i in range(1, l - 1):
        if dl[i - 1][1] < dl[i][1] and dl[i][1] >= dl[i + 1][1]:
            result.add(dl[i])
        if dl[i - 1][1] <= dl[i][1] and dl[i][1] > dl[i + 1][1]:
            result.add(dl[i])
    return result


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


mov_avg_window = 25


def get_val(y, coeff):
    return coeff[0] * y ** 2 + coeff[1] * y + coeff[2]


def get_curvature(coeff, y):
    A = coeff[0]
    B = coeff[1]
    R_curve = (1 + (2 * A * y + B) ** 2) ** 1.5 / 2 / A
    return R_curve


def detect_lane(img):
    img_size = img.shape
    steps = 8
    margin = 100
    minpix = 50

    histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)
    mean_lane = moving_average(histogram, mov_avg_window)

    peaks = find_peaks(mean_lane)
    left_peaks = filter(lambda x: x[0] <= 400, peaks)
    right_peaks = filter(lambda x: x[0] >= 800, peaks)

    leftx_base = sorted(left_peaks, key=lambda x: x[1], reverse=True)[0][0]
    rightx_base = sorted(right_peaks, key=lambda x: x[1], reverse=True)[0][0]

    img_zero = np.zeros_like(img)
    img_zero[:, :] = img[:, :]
    out_img = np.dstack((img, img, img))

    prev_left = leftx_base
    prev_right = rightx_base

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []

    for i in range(steps):
        leftx = prev_left
        rightx = prev_right

        y_high = int(img_size[0] - img_size[0] * i / steps)
        y_low = int(img_size[0] - img_size[0] * (i + 1) / steps)

        histogram = np.sum(img[y_low:y_high, :], axis=0)
        mean_lane = moving_average(histogram, mov_avg_window)

        peaks = find_peaks(mean_lane)

        left_candidates = list(filter(lambda x: abs(x[0] - leftx) <= 2 * margin, peaks))
        if len(left_candidates) > 0:
            leftx = sorted(left_candidates, key=lambda x: x[1], reverse=True)[0][0]

        right_candidates = list(filter(lambda x: abs(x[0] - rightx) <= 2 * margin, peaks))
        if len(right_candidates) > 0:
            rightx = sorted(right_candidates, key=lambda x: x[1], reverse=True)[0][0]

        xleft_low = leftx - margin
        xleft_high = leftx + margin
        xright_low = rightx - margin
        xright_high = rightx + margin

        cv2.rectangle(out_img, (xleft_low, y_low), (xleft_high, y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (xright_low, y_low), (xright_high, y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= xleft_low) & (
            nonzerox < xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= xright_low) & (
            nonzerox < xright_high)).nonzero()[0]

        if len(good_left_inds) > minpix:
            leftx = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        prev_left = leftx
        prev_right = rightx

    # done iterations

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 10 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 10 else None

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, left_fit, right_fit
