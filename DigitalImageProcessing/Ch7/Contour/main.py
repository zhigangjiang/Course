#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/11/24 17:11
# @Author  : ZhigangJiang
# @File    : main.py
# @Software: PyCharm
# @Description:

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plt_show_opcv(title, image):
    if image.shape.__len__() == 3:
        plt.imshow(image[:, :, ::-1])
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def pme(titles, images, rc=None):
    row = None
    col = None
    if rc is None:
        length = titles.__len__()
        row = int(np.sqrt(length))
        col = int(length / row)
        if length - row - col > 0:
            row += 1
    else:
        row = rc[0]
        col = rc[1]

    for i in range(titles.__len__()):
        plt.subplot(row, col, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([i])
    plt.show()


def contour_morphology(image):
    return image - cv2.erode(image, np.ones((3, 3), np.uint8))


def close_(img, kernel):
    dilate = cv2.dilate(img, kernel)
    erosion = cv2.erode(dilate, kernel)
    return erosion


def open_(img, kernel):
    erosion = cv2.erode(img, kernel)
    dilate = cv2.dilate(erosion, kernel)
    return dilate


house = cv2.imread("images/10_16_a.jpg", 0)
plt_show_opcv("house", house)

mask_averaging = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], np.float)/25
house_averaging = cv2.filter2D(house, -1, mask_averaging)
plt_show_opcv("house_averaging", house_averaging)


house_contour_morphology = contour_morphology(house_averaging)
plt_show_opcv("house_contour_morphology", house_contour_morphology)

ret1, house_binary = cv2.threshold(house, 175, 255, cv2.THRESH_BINARY)
plt_show_opcv("house_binary", house_binary)

house_close = close_(house_binary, np.ones((5, 5), np.uint8))
plt_show_opcv("house_close", house_close)


pme(["house", "house_averaging", "house_contour_morphology"],
    [house, house_averaging, house_contour_morphology])


# mask_45 = np.array([
#     [-1, -1,  2],
#     [-1,  2, -1],
#     [2,  -1, -1],
# ])
#
# house_segment_45 = cv2.filter2D(house_contour_morphology, -1, mask_45)
# plt_show_opcv("house_segment_45", house_segment_45)
#
# mask__45 = np.array([
#     [2, -1, -1],
#     [-1, 2, -1],
#     [-1, -1, 2],
# ])
# house_segment__45 = cv2.filter2D(house_contour_morphology, -1, mask__45)
# plt_show_opcv("house_segment_-45", house_segment__45)
#
#
# mask_0 = np.array([
#     [-1, -1, -1],
#     [2,   2,  2],
#     [-1, -1, -1],
# ])
# house_segment_0 = cv2.filter2D(house_contour_morphology, -1, mask_0)
# plt_show_opcv("house_segment_0", house_segment_0)


# pme(["house_contour_morphology", "house_segment_45",
#      "house_segment__45", "house_segment_0"],
#     [house_contour_morphology, house_segment_45,
#      house_segment__45, house_segment_0])



mask_sobel_horizontal = np.array([
    [-1, -2, -1],
    [0,   0,  0],
    [1,   2,  1],
])
mask_sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])

mask_sobel_diagonal = np.array([
    [-2, -1, 0],
    [-1, 0,  1],
    [0,  1,  2],
])


house_sobel_horizontal = cv2.filter2D(house_contour_morphology, -1, mask_sobel_horizontal)
plt_show_opcv("house_sobel_horizontal", house_sobel_horizontal)

house_sobel_vertical = cv2.filter2D(house_contour_morphology, -1, mask_sobel_vertical)
plt_show_opcv("house_sobel_vertical", house_sobel_vertical)

house_sobel_diagonal = cv2.filter2D(house_contour_morphology, -1, mask_sobel_diagonal)
plt_show_opcv("house_sobel_diagonal", house_sobel_diagonal)

house_sobel = house_sobel_horizontal + house_sobel_vertical + house_sobel_diagonal
plt_show_opcv("house_sobel", house_sobel)

pme(["house_sobel_horizontal", "house_sobel_vertical",
     "house_sobel_diagonal", "house_sobel"],
    [house_sobel_horizontal, house_sobel_vertical,
     house_sobel_diagonal, house_sobel])


#
# edges = cv2.Canny(house_averaging, 50, 50)
# plt_show_opcv('edges', edges)
# re = np.zeros(house.shape, np.uint8)
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=5)
# lines = lines[:, 0, :]
# for x1, y1, x2, y2 in lines:
#     cv2.line(re, (x1, y1), (x2, y2), (255, 255, 255), 1)
#
# plt_show_opcv('re', re)
# pme(["edges", "re"],
#     [edges, re])
#


