#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 21:07
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
        l = titles.__len__()
        row = int(np.sqrt(l))
        col = int(l / row)
    else:
        row = rc[0]
        col = rc[1]

    for i in range(titles.__len__()):
        plt.subplot(row, col, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([i])
    plt.show()


def open_(img, kernel):
    erosion = cv2.erode(img, kernel)
    dilate = cv2.dilate(erosion, kernel)
    return dilate


def close_(img, kernel):
    dilate = cv2.dilate(img, kernel)
    erosion = cv2.erode(dilate, kernel)
    return erosion

def sp_noisy(image, s_vs_p=0.5, amount=0.08):
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out


img = cv2.imread('images/12.png', 0)
ret, img_threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
pme(["img", "img_threshold"], [img, img_threshold])

# kernel1 = np.ones((1, 200), np.uint8)
# kernel2 = np.ones((90, 1), np.uint8)
# erosion = cv2.erode(img_threshold, kernel1)
# erosion2 = cv2.erode(img_threshold, kernel2)
# pme(["img_threshold", "erosion1", "erosion2"], [img_threshold, erosion, erosion2])
#
# kernel1 = np.ones((1, 200), np.uint8)
# kernel2 = np.ones((90, 1), np.uint8)
# dilate1 = cv2.dilate(erosion, kernel1)
# dilate2 = cv2.dilate(erosion, kernel2)
# pme(["img_threshold", "dilate1", "dilate2"], [img_threshold, dilate1, dilate2])

# kernel1 = np.ones((100, 100), np.uint8)
# open_res = open_(img_threshold, kernel1)
# close_res = close_(img_threshold, kernel1)
#
# pme(["img_threshold", "open_res", "close_res"], [img_threshold, open_res, close_res])

img_s = sp_noisy(img_threshold, 1)

img_p = sp_noisy(img_threshold, 0)

kernel1 = np.ones((10, 10), np.uint8)
open_res = open_(img_p, kernel1)
close_res = close_(img_s, kernel1)
pme(["img_salt", "img_pepper", "close_res", "open_res"],
    [img_s, img_p, open_res, close_res])