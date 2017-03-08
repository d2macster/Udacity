import numpy as np
import cv2

# src = np.float32([
#     [580, 460],
#     [700, 460],
#     [1040, 680],
#     [260, 680],
# ])
#
# dst = np.float32([
#     [260, 0],
#     [1040, 0],
#     [1040, 720],
#     [260, 720],
# ])

img_w = 1280
img_h = 720
img_w2 = img_w / 2


src = np.float32([
    [img_w2 - 75, 460],
    [img_w2 + 100, 460],
    [img_w2 + 680, 680],
    [img_w2 - 500, 680],
])

dst = np.float32([
    [0, 0],
    [img_w, 0],
    [img_w, img_h],
    [0, img_h],
])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def add_warp_points(img):
    mask = np.zeros_like(img).astype(np.uint8)
    cv2.fillPoly(mask, np.int_([src]), (0, 255, 0))
    result = cv2.addWeighted(img, 1, mask, 0.3, 0)
    return result


def warp(img):
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


def unwarp(img):
    return cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
