#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 12:39
# @Author  : ZhigangJiang
# @File    : main.py
# @Software: PyCharm
# @Description:hit and miss translate

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


def hmt(a, b):
    b1 = ~b
    b2 = b
    a1 = ~a
    a2 = a
    pme(["b1", "b2", "x1", "x2"],
        [b1, b2, a1, a2])
    x1_erode_b1 = cv2.erode(a1, b1)
    x2_erode_b2 = cv2.erode(a2, b2)
    plt_show_opcv("a1_erode_b1", x1_erode_b1)
    plt_show_opcv("a1_erode_b1_", cv2.dilate(x1_erode_b1, np.ones((10, 10), np.uint8)))
    plt_show_opcv("a2_erode_b2", x2_erode_b2)
    plt_show_opcv("a2_erode_b2_", cv2.dilate(x2_erode_b2, np.ones((10, 10), np.uint8)))
    r = cv2.bitwise_and(x1_erode_b1, x2_erode_b2)
    return r


def contour(image):
    return image - cv2.erode(image, np.ones((3, 3), np.uint8))


image_X = cv2.imread("images/X.png", 0)
image_B = cv2.imread("images/B_triangle.png", 0)

ret1, image_X = cv2.threshold(image_X, 127, 255, cv2.THRESH_BINARY)
ret2, image_B = cv2.threshold(image_B, 127, 255, cv2.THRESH_BINARY)

plt_show_opcv("contour", contour(image_X))

plt_show_opcv("X", image_X)
plt_show_opcv("B", image_B)
re = hmt(image_X, image_B)

targets = []
for i in range(re.shape[0]):
    for j in range(re.shape[1]):
        if re[i][j]:
            targets.append((j, i))
            print(i, j)

for target in targets:
    image_X = cv2.drawMarker(image_X, target, 125, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
plt_show_opcv("re", image_X)
