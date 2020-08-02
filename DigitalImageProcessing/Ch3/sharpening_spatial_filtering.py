#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 23:18
# @Author  : ZhigangJiang
# @File    : sharpening_spatial_filtering.py
# @Software: PyCharm
# @Description: sharpening spatial filtering
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plt_show_opcv(title, image, cv=False):
    if image.shape.__len__() == 3:
        plt.imshow(image[:, :, ::-1])
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
    if cv:
        cv2.imshow(title, image)
        cv2.waitKey(0)


def spatial_filtering(image, kernel, filter_):
    out = np.copy(image)
    h = image.shape[0]
    w = image.shape[1]
    for x in range(h):
        # print(str(int(x/h * 100)) + "%")
        for y in range(w):
            filter_(image, x, y, kernel, out)
    return out


def linear_filter(image, x, y, kernel, out):
    sum_wf = 0
    m = kernel.shape[0]
    n = kernel.shape[1]
    a = int((m - 1) / 2)
    b = int((n - 1) / 2)
    for s in range(-a, a + 1):
        for t in range(-b, b + 1):
            # convolution rotation 180
            x_s = (x - s) if (x - s) in range(0, image.shape[0] - 1) else 0
            y_t = (y - t) if (y - t) in range(0, image.shape[1] - 1) else 0
            sum_wf += kernel[a + s][b + t] * image[x_s][y_t]
    if sum_wf < 0:
        sum_wf = 0
    if sum_wf > 255:
        sum_wf = 255
    out[x][y] = int(sum_wf)


leaf = cv2.imread("images/3_38_a.jpg")
leaf = cv2.cvtColor(leaf, cv2.COLOR_RGB2GRAY)
leaf = cv2.resize(leaf, (400, 400))
plt_show_opcv("leaf", leaf)

laplacian_mask1 = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0],
])

laplacian_mask2 = np.array([
    [1,  1, 1],
    [1, -8, 1],
    [1,  1, 1],
])

laplacian_mask3 = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1],
])


gradient_mask = np.array([
    [0,  0,  0],
    [0, -2,  1],
    [0,  1,  0],
])


image_laplacian_mask1 = spatial_filtering(leaf, laplacian_mask1, linear_filter)
plt_show_opcv("image_laplacian_mask1", image_laplacian_mask1)

image_laplacian_mask2 = spatial_filtering(leaf, laplacian_mask2, linear_filter)
plt_show_opcv("image_laplacian_mask2", image_laplacian_mask2)


image_laplacian_mask3 = spatial_filtering(leaf, laplacian_mask3, linear_filter)
plt_show_opcv("image_laplacian_mask1", image_laplacian_mask3)


leaf_gradient_mask = spatial_filtering(leaf, gradient_mask, linear_filter)
plt_show_opcv("gradient_mask", leaf_gradient_mask)
