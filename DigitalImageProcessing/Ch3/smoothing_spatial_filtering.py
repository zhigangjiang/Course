#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 22:39
# @Author  : ZhigangJiang
# @File    : smoothing_spatial_filtering.py
# @Software: PyCharm
# @Description:smoothing spatial filtering
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plt_show_opcv(title, image):
    if image.shape.__len__() == 3:
        plt.imshow(image[:, :, ::-1])
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def sp_noisy(image, s_vs_p=0.5, amount=0.08):
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out


def spatial_filtering(image, kernel, filter_):
    out = np.copy(image)
    h = image.shape[0]
    w = image.shape[1]
    for x in range(h):
        print(str(int(x/h * 100)) + "%")
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
    out[x][y] = sum_wf


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


def nonlinear_max_filter(image, x, y, kernel, out):
    sp = []
    m = kernel.shape[0]
    n = kernel.shape[1]
    a = int((m - 1) / 2)
    b = int((n - 1) / 2)
    for s in range(-a, a + 1):
        for t in range(-b, b + 1):
            # convolution rotation 180
            x_s = (x - s) if (x - s) in range(0, image.shape[0] - 1) else 0
            y_t = (y - t) if (y - t) in range(0, image.shape[1] - 1) else 0
            if kernel[a + s][b + t]:
                sp.append(image[x_s][y_t])
    out[x][y] = np.max(sp)


def nonlinear_min_filter(image, x, y, kernel, out):
    sp = []
    m = kernel.shape[0]
    n = kernel.shape[1]
    a = int((m - 1) / 2)
    b = int((n - 1) / 2)
    for s in range(-a, a + 1):
        for t in range(-b, b + 1):
            # convolution rotation 180
            x_s = (x - s) if (x - s) in range(0, image.shape[0] - 1) else 0
            y_t = (y - t) if (y - t) in range(0, image.shape[1] - 1) else 0
            if kernel[a + s][b + t]:
                sp.append(image[x_s][y_t])
    out[x][y] = np.min(sp)


leaf = cv2.imread("images/leaf.jpg")
leaf = cv2.cvtColor(leaf, cv2.COLOR_RGB2GRAY)
leaf = cv2.resize(leaf, (400, 400))
plt_show_opcv("leaf", leaf)

leaf_sp_nose = sp_noisy(leaf, 0.5, 0.08)
plt_show_opcv("leaf_sp_nose", leaf_sp_nose)

k1 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], np.float32)

k2 = np.ones((5, 5))

k3 = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
])

leaf_smooth_averaging = spatial_filtering(leaf_sp_nose, k1/(k1.shape[0] * k1.shape[1]), linear_filter)
plt_show_opcv("leaf_smooth_averaging", leaf_smooth_averaging)

leaf_smooth_median = spatial_filtering(leaf_sp_nose, k1, nonlinear_median_filter)
plt_show_opcv("leaf_smooth_median1", leaf_smooth_median)
leaf_smooth_median = spatial_filtering(leaf_sp_nose, k2, nonlinear_median_filter)
plt_show_opcv("leaf_smooth_median2", leaf_smooth_median)
leaf_smooth_median = spatial_filtering(leaf_sp_nose, k3, nonlinear_median_filter)
plt_show_opcv("leaf_smooth_median3", leaf_smooth_median)

leaf_smooth_max = spatial_filtering(leaf_sp_nose, k1, nonlinear_max_filter)
plt_show_opcv("leaf_smooth_max", leaf_smooth_max)

leaf_smooth_min = spatial_filtering(leaf_sp_nose, k1, nonlinear_min_filter)
plt_show_opcv("leaf_smooth_min", leaf_smooth_min)
