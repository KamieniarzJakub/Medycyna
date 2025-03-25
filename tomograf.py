import cv2
import numpy as np
import pydicom


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


dcm_data = pydicom.dcmread('img-dicom/Kropka.dcm')
print(dcm_data)
print("PatientName" in dcm_data.dir())
dcm_data.PatientBirthDate = "13023"
print(dcm_data.PatientBirthDate)

img = dcm_data.pixel_array
print(img)
dcm_data.PatientName = "Marek Marek"
pydicom.dcmwrite("output/test.dicom", dcm_data)

# DICOM
# a) Podstawowe informacje o pacjencie
# ID pacjenta -> PatientID
# imie i nazwisko -> PatientName
# wiek -> PatientAge
# płeć -> PatientSex
# data urodzenia -> PatientBirthDate

# b) Data badania -> StudyDate

# c) Komentarzy -> StudyComments lub ImageComments

# obraz -> pixel_array 

#img = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
img = cv2.copyMakeBorder(
    img,
    int(max((w - h) / 2, 0)),
    int(max((w - h) / 2, 0)),
    int(max((h - w) / 2, 0)),
    int(max((h - w) / 2, 0)),
    cv2.BORDER_CONSTANT,
    value=[0],
)
cv2.imshow("source", img)


def radon(
    img,
    size=None,
    angle_step=2,
    n=180,
    emiters_angles=180,
    inverse=False,
):
    # n - Liczba emiterów/detetektorów
    # emiters_angles = 180  # Kąt rozposzenia emiterów
    if size is None:
        size = img.shape
    h, w = size
    r = w // 2
    angles = np.arange(0, 180, angle_step)
    if inverse:
        out = np.zeros(size)
    else:
        out = np.zeros((len(angles), n))

    for a, angle in enumerate(angles):
        detectors_angles = np.linspace(
            angle - emiters_angles // 2, angle + emiters_angles // 2, n
        )
        for i, emiter_angle in enumerate(detectors_angles):
            emiter_angle_rad = np.deg2rad(emiter_angle)
            detector_angle_rad = np.deg2rad(detectors_angles[n - 1 - i])
            x_e = min(r - int(r * np.cos(emiter_angle_rad)), w - 1)
            x_d = min(r - int(r * np.cos(np.pi + detector_angle_rad)), w - 1)
            y_e = min(r - int(r * np.sin(emiter_angle_rad)), h - 1)
            y_d = min(r - int(r * np.sin(np.pi + detector_angle_rad)), h - 1)
            # img[y_e][x_e] = 255
            # img[y_d][x_d] = 255

            points = bresenham_line(x_e, y_e, x_d, y_d)
            for x, y in points:
                if inverse:
                    out[y][x] += img[a][i]
                else:
                    out[a][i] += img[y][x]
                # img[y][x] = 100

    out /= out.max()
    return out


sinogram = radon(img)
cv2.imshow("sinogram", sinogram)

# Odwrotny Radon
tomograf = radon(sinogram, inverse=True, size=img.shape)


cv2.imshow("tomograf", tomograf)
cv2.waitKey()
