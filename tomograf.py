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

img = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
img = cv2.copyMakeBorder(img, max((w-h)//2, 0), max((w-h)//2, 0), max((h-w)//2, 0), max((h-w)//2, 0), cv2.BORDER_CONSTANT, value=[0])
h, w = img.shape
print(img.shape)
angle_step = 2
n = 180 #Liczba emiterów/detetektorów
emiters_angles = 30  #Kąt rozposzenia emiterów

angles = np.arange(0, 180, angle_step)
sinogram = np.zeros((len(angles), n))

r = w//2
max_value = 0
for a, angle in enumerate(angles):
    detectors_angles = np.linspace(angle-emiters_angles//2, angle+emiters_angles//2, n)
    for i, emiter_angle in enumerate(detectors_angles):
        emiter_angle_rad = np.deg2rad(emiter_angle)
        detector_angle_rad = np.deg2rad(detectors_angles[n-1-i])
        x_e = min(r-int(r * np.sin(emiter_angle_rad)), w-1)
        x_d = min(r-int(r * np.sin(np.pi+detector_angle_rad)), w-1)
        y_e = min(r-int(r * np.cos(emiter_angle_rad)), h-1)
        y_d = min(r-int(r * np.cos(np.pi+detector_angle_rad)), h-1)
        #img[y_e][x_e] = 255
        #img[y_d][x_d] = 255

        for x,y in bresenham_line(x_e, y_e, x_d, y_d):
            sinogram[a][i]+=img[y][x]
            #img[y][x] = 100
        if sinogram[a][i]>max_value:
            max_value = sinogram[a][i]
sinogram/=max_value
sinogram*=255
#print(sinogram[0][90], max_value)

cv2.imshow("sinogram", sinogram)

cv2.imshow("tomograf", img)
cv2.waitKey()
