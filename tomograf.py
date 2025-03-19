import cv2
import numpy as np

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

img = cv2.imread("img/Kolo.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
img = cv2.copyMakeBorder(img, max((w-h)//2, 0), max((w-h)//2, 0), max((h-w)//2, 0), max((h-w)//2, 0), cv2.BORDER_CONSTANT, value=[0])
h, w = img.shape
print(img.shape)

angle_step = 0
n = 10
emiters_angles = 45
emiters_angles = np.linspace(angle_step, angle_step + emiters_angles, n)

r = w//2
for angle in emiters_angles:
    angle_rad = np.deg2rad(angle)
    x_e = r-int(r * np.sin(angle_rad))
    y_e = r-int(r * np.cos(angle_rad))
    img[y_e][x_e] = 255
    img[h-y_e-1][w-x_e-1] = 255
    for x,y in bresenham_line(x_e, y_e, h-x_e-1, w-y_e-1):
        img[y][x] = 100

cv2.imshow("tomograf", img)
cv2.waitKey()
