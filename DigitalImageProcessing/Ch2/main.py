# -*- coding: utf-8 -*-

"""
    @Date: 2019-05-14
    @Written By: jiangzhigang
    @Description:
    1. 计算累计概率
    2. 每个像素映射 - 灰度值越大 - 255所乘概率越大 - 均衡作用
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
 

def equalize_hist(img, level):
    img = cv2.copyTo(img, None)
    height = img.shape[0]
    width = img.shape[1]

    color_gray = np.zeros(level, np.float)
    # n
    for i in range(height):
        for j in range(width):
            color_gray[img[i, j]] += 1
    # pr
    are = height * width
    for i in range(level):
        color_gray[i] /= are
    # sum(pr)
    for i in range(1, level):
        color_gray[i] += color_gray[i - 1]
    # sk
    for i in range(level):
        color_gray[i] = color_gray[i] * (level - 1) + 0.5
    # r -> s
    for i in range(height):
        for j in range(width):
            img[i, j] = color_gray[img[i, j]]
    return img


def show_hist(img, level, title="hist"):
    height = img.shape[0]
    width = img.shape[1]

    color_gray = np.zeros(level, np.float)
    # 遍历图片
    for i in range(height):
        for j in range(width):
            color_gray[img[i, j]] += 1

    # 概率
    are = height * width
    for i in range(level):
        color_gray[i] /= are

    x = np.linspace(0, level-1, level)
    y = color_gray
    plt.bar(x, y, 1, alpha=1, color='b')
    plt.title(title)
    plt.show()


img0 = cv2.imread("images/leaf.jpg", 0)
gray = cv2.resize(img0, (400, 400))
# test use
# gray = cv2.resize(img0, (4, 4))
# gray[0][0] = 0
# gray[0][1] = 1
# gray[0][2] = 2
# gray[0][3] = 3
# gray[1][0] = 4
# gray[1][1] = 4
# gray[1][2] = 4
# gray[1][3] = 4
# gray[2][0] = 5
# gray[2][1] = 5
# gray[2][2] = 5
# gray[2][3] = 5
# gray[3][0] = 6
# gray[3][1] = 7
# gray[3][2] = 5
# gray[3][3] = 4

plt.imshow(gray, cmap='gray')
plt.title('resource')
plt.show()

# 原图
show_hist(gray, 256, "rec_hist")
# 均衡化
destination = equalize_hist(gray, 256)
plt.imshow(destination, cmap='gray')
plt.title('destination')
plt.show()
show_hist(destination, 256, 'dst_hist')

# opencv自带 均衡化
destination_ = cv2.equalizeHist(gray)
plt.imshow(destination_, cmap='gray')
plt.title('destination_')
plt.show()
show_hist(destination_, 256, 'op_cv_hist')
cv2.waitKey(0)
