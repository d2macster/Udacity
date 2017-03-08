import cv2
import numpy as np


hls_low = np.array([0, 0, 100])
hls_high = np.array([255, 255, 255])

white_hsv_low = np.array([0, 0, 160])
white_hsv_high = np.array([255, 80, 255])

yellow_hsv_low = np.array([0, 100, 100])
yellow_hsv_high = np.array([80, 255, 255])


def get_sobel(img, sobel_kernel=5):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    return sobelx, sobely


def dir_threshold(sobelx, sobely, thresh=(0.7, 1.2)):
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def abs_sobel_thresh(sobel, sobel_thresh=(25, 200)):
    abs_sobel = np.absolute(sobel)
    scale_factor = 255.0 / np.max(abs_sobel)

    scaled_sobel = np.uint8(scale_factor * abs_sobel)
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    return sbinary


def mag_thresh(sobelx, sobely, mag_thresh=(20, 150)):
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = 255.0 / np.max(gradmag)
    gradmag = (scale_factor * gradmag).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def color_mask(img, low, high):
    # Return mask from HSV
    mask = cv2.inRange(img, low, high)
    return mask


def apply_color_mask(hsv, img, low, high):
    # Apply color mask to image
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def hls_thresh(img, ):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = color_mask(hls, hls_low, hls_high)
    return mask


def yellow_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = color_mask(hsv, yellow_hsv_low, yellow_hsv_high)
    return mask


def white_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = color_mask(hsv, white_hsv_low, white_hsv_high)
    return mask

def get_thresholded_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    w_mask = color_mask(hsv, white_hsv_low, white_hsv_high)
    y_mask = color_mask(hsv, yellow_hsv_low, yellow_hsv_high)

    return cv2.bitwise_or(w_mask, y_mask)

def get_thresholded_hsl(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_s = hls[:,:,2]
    hls_l = hls[:,:,1]

    s_sobelx, s_sobely = get_sobel(hls_s)
    s_gradx = abs_sobel_thresh(sobel=s_sobelx)
    s_grady = abs_sobel_thresh(sobel=s_sobely)
    s_mag_t = mag_thresh(sobelx=s_sobelx, sobely=s_sobely)
    s_dir_t = dir_threshold(sobelx=s_sobelx, sobely=s_sobely)

    mask_2 = cv2.bitwise_or(s_gradx, s_grady)

    l_sobelx, l_sobely = get_sobel(hls_l)
    l_gradx = abs_sobel_thresh(sobel=l_sobelx)
    l_grady = abs_sobel_thresh(sobel=l_sobely)
    l_mag_t = mag_thresh(sobelx=l_sobelx, sobely=l_sobely)
    l_dir_t = dir_threshold(sobelx=l_sobelx, sobely=l_sobely)

    mask_3 = cv2.bitwise_or(l_gradx, l_grady)

    mask_4 = cv2.bitwise_or(mask_2, mask_3)

    return mask_4
