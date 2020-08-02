#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 22:24
# @Author  : ZhigangJiang
# @File    : as.py
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


def bhpf(image, d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    transfor_matrix = np.zeros(image.shape)
    M = transfor_matrix.shape[0]
    N = transfor_matrix.shape[1]
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            filter_mat = 1 / (1 + np.power(d / D, 2))
            transfor_matrix[u, v] = filter_mat
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * transfor_matrix)))
    return new_img


img = cv2.imread('4_29_a.jpg', 0)
plt_show_opcv("image", img)

bhpf_re = bhpf(img, 150)

plt_show_opcv("bhpf_re", bhpf_re)