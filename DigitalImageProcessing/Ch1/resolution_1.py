#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 21:56
# @Author  : ZhigangJiang
# @File    : resolution_1.py
# @Software: PyCharm
# @Description: changing spatial resolution

import cv2
import numpy as np
import math


def nearest(img, size):
    """
    Nearest neighbor interpolation
    :param img: source image
    :param size: (height, width)
    :return: destination image
    """
    re = np.zeros([size[0], size[1], 1], np.uint8)
    for x in range(size[0]):
        for y in range(size[1]):
            new_x = int(x * (img.shape[0] / size[0]))
            new_y = int(y * (img.shape[1] / size[1]))
            re[x, y] = img[new_x, new_y]
    return re


def bilinear(img, size):
    """
    Bilinear interpolation
    :param img: source image
    :param size: (height, width)
    :return: destination image
    """
    re = np.zeros([size[0], size[1], 1], np.uint8)
    for x in range(size[0]):
        for y in range(size[1]):
            new_x = x * (img.shape[0] / size[0])
            new_y = y * (img.shape[1] / size[1])
            i = int(new_x)
            j = int(new_y)

            u = new_x - i
            v = new_y - j

            if i + 1 >= img.shape[0]:
                i = img.shape[0] - 2
            if j + 1 >= img.shape[1]:
                j = img.shape[1] - 2

            # f(i+u,j+v)=(1−u)(1−v)f(i,j)+(1−u)vf(i,j+1)+u(1−v)f(i+1,j)+uvf(i+1,j+1)
            # 小数越大，距离越远，所以该点发挥的作用更小些所以用1-u和1-v来乘i,j坐标
            re[x, y] = (1-u)*(1-v)*img[i, j] + (1-u)*v*img[i, j+1] + u*(1-v)*img[i+1, j] + u*v*img[i+1, j+1]
    return re


image = cv2.imread("images/2_20_a.jpg", cv2.IMREAD_UNCHANGED)
#
size_1 = (int(image.shape[0]*0.2), int(image.shape[1]*0.2))
size_2 = (image.shape[0], image.shape[1])
img_n = nearest(image, size_1)
img_n = nearest(img_n, size_2)

img_b = bilinear(image, size_1)
img_b = bilinear(img_b, size_2)

# OpenCV
img_o = cv2.resize(image, (size_1[1], size_1[0]))
img_o = cv2.resize(img_o, (size_2[1], size_2[0]))

cv2.imshow("image", image)
# cv2.imwrite("images/full.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
cv2.imshow("opencv", img_o)
cv2.imshow("nearest", img_n)
cv2.imshow("bilinear", img_b)


cv2.waitKey(0)