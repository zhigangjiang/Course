#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 7:57
# @Author  : ZhigangJiang
# @File    : main.py
# @Software: PyCharm
# @Description:

import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


def plt_show_opcv(title, image):
    if image.shape.__len__() == 3:
        plt.imshow(image[:, :, ::-1])
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def prue(image):
    out = np.copy(image)
    for x in range(image.shape[0]):
        # print(str(int(x / image.shape[0] * 100)) + "%")
        for y in range(image.shape[1]):
            b, g, r = image[x][y]
            b = 255 if b > 127 else 0
            g = 255 if g > 127 else 0
            r = 255 if r > 127 else 0
            out[x][y] = b, g, r
    return out

def bgr_2_hsi(image):
    """
    :param image: RGB model image
    :return: HSI model image and slicing image in H
    """
    out = np.copy(image)
    out_slicing = np.zeros(image.shape, np.uint8)

    for x in range(image.shape[0]):
        # print(str(int(x / image.shape[0] * 100)) + "%")
        for y in range(image.shape[1]):
            b, g, r = image[x][y]
            b, g, r = int(b), int(g), int(r)
            i_s = np.sum([b, g, r])
            i = i_s / 3
            # i == 0, s and h is no sense

            if i_s == 0:
                i = 0
                s = 0
                h = 0
                out[x][y] = h, s, i
                continue

            s = (1 - (3 * np.min([b, g, r])) / i_s) * 255

            # s == 0 h is no sense
            if s == 0:
                h = 0
                out[x][y] = h, s, i
                continue

            thea = np.arccos((2 * r - g - b) / (2 * np.sqrt((r - g) ** 2 + (r - b) * (g - b))))
            if g >= b:
                h1 = thea
            else:
                h1 = np.pi * 2 - thea
            h1 = np.rad2deg(h1)
            # slicing
            if (int(h1) in range(0, 11) or int(h1) in range(350, 361) ) and s/255 > 0.1:
                print(int(h1))
                out_slicing[x][y] = image[x][y]

            h = h1 / 360 * 255
            out[x][y] = h, s, i

    return out, out_slicing


def slicing(image):
    out = np.copy(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if int(image[x][y]/255*360) not in range(0, 30) or range(330, 360):
                out[x][y] = 255
    plt_show_opcv("out1", out)
    return out



def hsi_2_bgr(image):
    out = np.copy(image)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            h, s, i = image[x][y]
            h, s, i = h / 255 * 360, s / 255, i / 255
            b, g, r = 0, 0, 0
            # not use float in range(int, int) :(
            if h >= 0 and  h < 120:  # RG
                b = i * (1 - s)
                r = i * (1 + (s * math.cos(math.radians(h)) / math.cos(math.radians(60 - h))))
                g = 3 * i - (b + r)
            elif h >= 120 and  h < 240:  # GB
                h -= 120
                r = i * (1 - s)
                g = i * (1 + (s * math.cos(math.radians(h)) / math.cos(math.radians(60 - h))))
                b = 3 * i - (r + g)
            elif h >= 240 and  h < 360:  # BR
                h -= 240
                g = i * (1 - s)
                b = i * (1 + (s * np.cos(math.radians(h)) / np.cos(math.radians(60 - h))))
                r = 3 * i - (g + b)

            out[x][y] = b * 255, g * 255, r * 255
    return out

def color_slicing(image, center, w):
    """
    :param image:
    :param center: g, b, r ib range 0 ~ 255
    :param w: width
    :return:
    """
    out = np.zeros(image.shape, np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            r_b, r_g, r_r = center
            a_b, a_g, a_r = image[x][y]
            if abs(r_b - a_b) < w/2 and abs(r_g - a_g) < w/2 and abs(r_r - a_r) < w/2:
                out[x][y] = image[x][y]
    return out

image = cv2.imread("images/6_16_a.jpg", 1)
# image = cv2.resize(image, (400, 400))
plt_show_opcv("image", image)
image = prue(image)
plt_show_opcv("image_prue", image)

image_hsi, image_slicing = bgr_2_hsi(image)
plt_show_opcv("image_red", image_slicing)

plt_show_opcv("image_hsi", image_hsi)
h, s, i = cv2.split(image_hsi)

image_cloor_slicing = color_slicing(image, (0.1922 *255, 0.1608 *255, 0.7863 * 255), 0.4549*255)
plt_show_opcv("image_cloor_slicing", image_cloor_slicing)

# plt_show_opcv("image_hsi_h", np.array(h))
# plt_show_opcv("image_hsi_s", np.array(s))
# plt_show_opcv("image_hsi_i", np.array(i))

image_hsi_2_bgr = hsi_2_bgr(cv2.merge([h, s, i]))
plt_show_opcv("image_hsi_2_bgr", image_hsi_2_bgr)

# image_hsi_slic_h = slicing(h)
# plt_show_opcv("image_hsi_slic_h", image_hsi_slic_h)
#
# image_hsi_slic_h_2_bgr = hsi_2_bgr(cv2.merge([image_hsi_slic_h, s,i]))
# plt_show_opcv("image_hsi_slic_h_2_bgr", image_hsi_slic_h_2_bgr)
