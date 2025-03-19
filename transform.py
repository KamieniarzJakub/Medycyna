import os
import cv2
import numpy as np
import base64
import io


# def get_radius(a, b, alpha):
#     return (
#         a * b / np.sqrt(np.power(a * np.sin(alpha), 2), np.power(b * np.cos(alpha), 2))
#     )


def rectangle_to_square(img):
    h, w = img.shape
    return cv2.copyMakeBorder(
        img,
        max((w - h) // 2, 0),
        max((w - h) // 2, 0),
        max((h - w) // 2, 0),
        max((h - w) // 2, 0),
        cv2.BORDER_CONSTANT,
        value=[0],
    )


def radon_transform(img):
    h, w, channels = img.shape
    delta_alpha = 10
    beam_num = 5
    E = (w // 2, h // 2)
    angles = np.arange(0, 360, delta_alpha)
    r = w / 2
    for angle in angles[: len(angles) // 2]:
        rad = np.deg2rad(angle)
        xe = int(r * (1 - np.cos(rad))) - 1
        ye = int(r * (1 - np.sin(rad))) - 1
        img[xe][ye] = (255, 0, 0)
        xdi = int(r * (1 - np.cos(rad + np.pi))) - 1
        ydi = int(r * (1 - np.sin(rad + np.pi))) - 1
        print(xe, ye, xdi, ydi)
        img[xdi][ydi] = (0, 255, 0)


my_img = cv2.imread("img/Kolo.jpg")
radon_transform(my_img)
cv2.imshow("img", cv2.resize(my_img, (1000, 1000), interpolation=cv2.INTER_LANCZOS4))
cv2.waitKey()
