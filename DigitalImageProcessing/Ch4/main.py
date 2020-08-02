#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 20:33
# @Author  : ZhigangJiang
# @File    : main.py
# @Software: PyCharm
# @Description:
import cv2
import numpy as np
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


def fft(image):
    f = np.fft.fft2(image)
    # move to center
    fshift = np.fft.fftshift(f)
    return fshift


def ifft(fshift):
    f1shift = np.fft.ifftshift(fshift)
    image_back = np.fft.ifft2(f1shift)
    image_back = np.abs(image_back)
    return image_back


def get_mask(shape, r):
    mask_ = np.zeros(shape, np.uint8)
    cv2.circle(mask_, (int(shape[1] / 2), int(shape[0] / 2)), r, 255, -1)
    return mask_


img = cv2.imread('4_29_a.jpg', 0)
plt_show_opcv("image", img)

fshift_re = fft(img)
show_re = np.log(np.abs(fshift_re))
plt_show_opcv("show_re", show_re)

mask = get_mask(img.shape, 5)
plt_show_opcv("mask", mask)
re = fshift_re * mask

new_img = ifft(re)
plt_show_opcv("image_back", new_img)

mask2 = get_mask(img.shape, 50)
plt_show_opcv("mask2", mask2)
re2 = fshift_re * mask2

new_img2 = ifft(re2)
plt_show_opcv("image_back2", new_img2)

mask3 = get_mask(img.shape, 150)
plt_show_opcv("mask3", mask3)
re3 = fshift_re * mask3

new_img3 = ifft(re3)
plt_show_opcv("image_back3", new_img3)
