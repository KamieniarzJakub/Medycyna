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
img = cv2.copyMakeBorder(img, int(max((w-h)/1.4, 0)), int(max((w-h)/1.4, 0)), int(max((h-w)//1.4, 0)), int(max((h-w)/1.4, 0)), cv2.BORDER_CONSTANT, value=[0])
h, w = img.shape
print(img.shape)
angle_step = 4
n = 360 #Liczba emiterów/detetektorów
emiters_angles = 180  #Kąt rozposzenia emiterów
angles = np.arange(0, 360, angle_step)
sinogram = np.zeros((len(angles), n))

r = w//2

#Radom
for a, angle in enumerate(angles):
    max_value = 0
    detectors_angles = np.linspace(angle-emiters_angles//2, angle+emiters_angles//2, n)
    for i, emiter_angle in enumerate(detectors_angles):
        emiter_angle_rad = np.deg2rad(emiter_angle)
        detector_angle_rad = np.deg2rad(detectors_angles[n-1-i])
        x_e = min(r-int(r * np.cos(emiter_angle_rad)), w-1)
        x_d = min(r-int(r * np.cos(np.pi+detector_angle_rad)), w-1)
        y_e = min(r-int(r * np.sin(emiter_angle_rad)), h-1)
        y_d = min(r-int(r * np.sin(np.pi+detector_angle_rad)), h-1)
        #img[y_e][x_e] = 255
        #img[y_d][x_d] = 255

        points = bresenham_line(x_e, y_e, x_d, y_d)
        for x,y in points:
            sinogram[a][i]+=img[y][x]
            #img[y][x] = 100
        
sinogram /= sinogram.max()

#print(sinogram[0][90], max_value)

cv2.imshow("sinogram", sinogram)

#Odwrotny Radom
tomograf = np.zeros(img.shape) #Rozmiar na suwaki
for a, angle in enumerate(angles):
    detectors_angles = np.linspace(angle-emiters_angles//2, angle+emiters_angles//2, n)
    for i, emiter_angle in enumerate(detectors_angles):
        emiter_angle_rad = np.deg2rad(emiter_angle)
        detector_angle_rad = np.deg2rad(detectors_angles[n-1-i])
        x_e = min(r-int(r * np.cos(emiter_angle_rad)), w-1)
        x_d = min(r-int(r * np.cos(np.pi+detector_angle_rad)), w-1)
        y_e = min(r-int(r * np.sin(emiter_angle_rad)), h-1)
        y_d = min(r-int(r * np.sin(np.pi+detector_angle_rad)), h-1)
        #img[y_e][x_e] = 255
        #img[y_d][x_d] = 255

        points = bresenham_line(x_e, y_e, x_d, y_d)
        for x,y in points:
            tomograf[y][x] += sinogram[a][i]
tomograf /= tomograf.max()

print(tomograf.max())
#tomograf -=tomograf.max()/4

cv2.imshow("tomograf", tomograf)
cv2.waitKey()
