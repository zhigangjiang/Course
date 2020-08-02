#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/04 20:53
# @Author  : ZhigangJiang
# @File    : resolution_1.py
# @Software: PyCharm
# @Description: reduce the intensity levels

import cv2


def reduce_intensity_levels(img, level):
    img = cv2.copyTo(img, None)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            si = img[x, y]
            # +0.5 四舍五入
            ni = int(level * si / 255 + 0.5) * (255 / level)
            img[x, y] = ni
    return img


a = 255/8
image = cv2.imread("images/2_20_a.jpg", cv2.IMREAD_UNCHANGED)
img_n = reduce_intensity_levels(image, 3)

cv2.imshow("image", image)
# cv2.imshow("opencv", img_o)
cv2.imshow("reduce_level", img_n)
cv2.waitKey(0)
