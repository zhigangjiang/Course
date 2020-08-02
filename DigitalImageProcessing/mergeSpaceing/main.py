#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/8 19:18
# @Author  : ZhigangJiang
# @File    : main.py
# @Software: PyCharm
# @Description: Combining Spatial Enhancement Methods
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


def pme(titles, images, rc=None):
    row = None
    col = None
    if rc is None:
        l = titles.__len__()
        row = int(np.sqrt(l))+1
        col = int(l / row)+1
    else:
        row = rc[0]
        col = rc[1]

    for i in range(titles.__len__()):
        plt.subplot(row, col, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([i])
    plt.show()


def spatial_filtering(image, kernel, filter_):
    out = np.copy(image)
    out = out.astype("int")
    image = image.astype("int")
    h = image.shape[0]
    w = image.shape[1]
    for x in range(h):
        print(str(int(x/h * 100)) + "%")
        for y in range(w):
            filter_(image, x, y, kernel, out)
    return out


def offset(image, num):
    out = np.copy(image)
    out = out.astype("int")
    image = image.astype("int")
    h = image.shape[0]
    w = image.shape[1]
    for x in range(h):
        print(str(int(x / h * 100)) + "%")
        for y in range(w):
            out[x, y] = image[x, y] + num
            if out[x, y] < 0:
                out[x, y] = 0

            if out[x, y] > 255:
                out[x, y] = 255
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


def nonlinear_median_filter(image, x, y, kernel, out):
    sp = []
    m = kernel.shape[0]
    n = kernel.shape[1]
    a = int((m - 1) / 2)
    b = int((n - 1) / 2)
    for s in range(-a, a + 1):
        for t in range(-b, b + 1):
            x_s = (x + s) if (x + s) in range(0, image.shape[0] - 1) else 0
            y_t = (y + t) if (y + t) in range(0, image.shape[1] - 1) else 0
            if kernel[a + s][b + t]:
                sp.append(image[x_s][y_t])
    out[x][y] = np.median(sp)


def nonlinear_filter_abs(image, x, y, kernel, out):
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

    sum_wf = abs(sum_wf)
    if sum_wf > 255:
        sum_wf = 255
    out[x][y] = int(sum_wf)


img = cv2.imread("images/3_43_a.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = img.astype("int")
# img = cv2.bitwise_not(img)

plt_show_opcv("img", img)

# Laplacian filtering
laplacian_mask = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1],
])

image_laplacian = spatial_filtering(img, laplacian_mask, linear_filter)
plt_show_opcv("image_laplacian", image_laplacian)
##

# Median filtering
median_mask = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])
# image_smooth_median = spatial_filtering(image_laplacian, median_mask, nonlinear_median_filter)
# plt_show_opcv("image_smooth_median", image_smooth_median)
# However, median filtering is a nonlinear process capable
# of removing image features.This is unacceptable in medical image processing.
#

# Soble filtering
soble_mask_horizontal = np.array([
    [-1, -2, -1],
    [0,   0,  0],
    [1,   2,  1],
])
soble_mask_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])


image_soble_horizontal = spatial_filtering(img, soble_mask_horizontal, nonlinear_filter_abs)
image_soble_vertical = spatial_filtering(img, soble_mask_vertical, nonlinear_filter_abs)
image_soble = image_soble_horizontal + image_soble_vertical
plt_show_opcv("image_soble", image_soble)

# Average filtering 5*5
average_mask = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
], np.float32)/25
image_average = spatial_filtering(image_soble, average_mask, linear_filter)
plt_show_opcv("image_average", image_average)

# Laplacian * [Average Soble]

laplacian_multiply_soble = cv2.multiply(image_laplacian, image_average)
x = np.max(laplacian_multiply_soble)/255
laplacian_multiply_soble = laplacian_multiply_soble/x
laplacian_multiply_soble = laplacian_multiply_soble.astype("int")
# laplacian_multiply_soble = spatial_filtering(laplacian_multiply_soble, median_mask, nonlinear_median_filter)
plt_show_opcv("laplacian_multiply_soble", laplacian_multiply_soble)

# [Laplacian * [Average Soble]] + Orgin
res = cv2.add(img, laplacian_multiply_soble)
plt_show_opcv("res", res)

# Power law transformation
image_power = res**0.5
plt_show_opcv("image_power", image_power)

pme(["img", "image_laplacian", "image_soble", "image_average", "laplacian_multiply_soble", "res", "image_power"],
    [img, image_laplacian, image_soble, image_average, laplacian_multiply_soble, res, image_power])
