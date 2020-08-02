#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 22:36
# @Author  : ZhigangJiang
# @File    : main.py.py
# @Software: PyCharm
# @Description:
import cv2
import numpy as np

freqs = np.fft.fftfreq(9, d=1. / 9).reshape(3, 3)
freqs2 = np.fft.ifftshift(np.fft.fftshift(freqs))
print()