import cv2
import numpy as np
import matplotlib.pyplot as plt

import scipy
# import scipy.stats


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


def add_gaussian_noise(image_in, noise_sigma=25):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image


def sp_noisy(image, s_vs_p=0.5, amount=0.08):
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out


def filter(image, op):
    new_image = np.zeros(image.shape)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = op(image[i - 1:i + 2, j - 1:j + 2])
    new_image = (new_image - np.min(image)) * (255 / np.max(image))
    return new_image.astype(np.uint8)


chip = cv2.imread("images/5_7_a.jpg", 0)
plt_show_opcv("chip", chip)

chip_gs = add_gaussian_noise(chip, 25)
plt_show_opcv("chip_gs", chip_gs)

chip_s = sp_noisy(chip, 1)
plt_show_opcv("chip_s", chip_s)

chip_p = sp_noisy(chip, 0)
plt_show_opcv("chip_p", chip_p)


chip_sp = sp_noisy(chip)
plt_show_opcv("chip_sp", chip_sp)


########################################

# 几何均值复原
def GeometricMeanOperator(roi):
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    re = p ** (1 / (roi.shape[0] * roi.shape[1]))
    if re < 0:
        re = 0
    if re > 255:
        re = 255
    return re


# chip_gm_gs = filter(chip_gs, GeometricMeanOperator)
# plt_show_opcv("chip_gm_gs", chip_gm_gs)
#
# chip_gm_s = filter(chip_s, GeometricMeanOperator)
# plt_show_opcv("chip_gm_s", chip_gm_s)
#
#
# chip_gm_p = filter(chip_p, GeometricMeanOperator)
# plt_show_opcv("chip_gm_p", chip_gm_p)


# 谐波均值滤波器
def HMeanOperator(roi):
    roi = roi.astype(np.float64)
    re = roi.shape[0] * roi.shape[1] / np.sum([1/(p+0.0001) for p in roi])
    if re < 0:
        re = 0
    if re > 255:
        re = 255
    return re


# chip_hm_gs = filter(chip_gs, HMeanOperator)
# plt_show_opcv("chip_hm_gs", chip_hm_gs)
#
# chip_hm_s = filter(chip_s, HMeanOperator)
# plt_show_opcv("chip_hm_s", chip_hm_s)
#
# chip_hm_p = filter(chip_p, HMeanOperator)
# plt_show_opcv("chip_hm_p", chip_hm_p)

def IHMeanOperator(roi, q):
    roi = roi.astype(np.float64)
    return np.mean(roi ** (q + 1)) / np.mean(roi ** q)


def IHMeanAlogrithm(image, q):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = IHMeanOperator(image[i - 1:i + 2, j - 1:j + 2], q)
    new_image = (new_image - np.min(image)) * (255 / np.max(image))
    return new_image.astype(np.uint8)


chip_ihm_gs = IHMeanAlogrithm(chip_gs, -1)
plt_show_opcv("chip_ihm_gs", chip_ihm_gs)

chip_ihm_s = IHMeanAlogrithm(chip_s, -1)
plt_show_opcv("chip_ihm_s", chip_ihm_s)

chip_ihm_p = IHMeanAlogrithm(chip_p, -1)
plt_show_opcv("chip_ihm_p", chip_ihm_p)
