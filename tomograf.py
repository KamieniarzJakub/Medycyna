import numpy as np
from numba import jit


@jit
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


@jit
def integer_bresenham(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    j = y1
    e = dy - dx

    for i in range(x1, x2):
        yield (i, j)
        if e >= 0:
            j += 1
            e -= dx
        i += 1
        e += dy


@jit
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
    angles = np.arange(0, 360, angle_step)
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

            points = bresenham_line(x_e, y_e, x_d, y_d)
            for x, y in points:
                if inverse:
                    out[y][x] += img[a][i]
                else:
                    out[a][i] += img[y][x]


@jit
def radon_step(angle_step, n, w, h):
    r = w / 2
    # n - Liczba emiterów/detetektorów
    # emiters_angles = 180  # Kąt rozposzenia emiterów
    detectors_angle = 90
    for a in range(0, 360, angle_step):
        for i in range(n):
            emiter_angle_rad = np.deg2rad(a)
            detector_angle_rad = np.deg2rad(
                a + np.pi - detectors_angle / 2 + i * (detectors_angle / n - 1)
            )
            x_e = min(int(r * np.cos(emiter_angle_rad)), w - 1)
            x_d = min(int(r * np.cos(detector_angle_rad)), w - 1)
            y_e = min(int(r * np.sin(emiter_angle_rad)), h - 1)
            y_d = min(int(r * np.sin(detector_angle_rad)), h - 1)

            for x, y in bresenham_line(x_e, y_e, x_d, y_d):
                yield (a, i, y, x)


# if __name__ == "__main__":
#     img = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
#     h, w = img.shape
#     img = cv2.copyMakeBorder(
#         img,
#         int(max((w - h) / 2, 0)),
#         int(max((w - h) / 2, 0)),
#         int(max((h - w) / 2, 0)),
#         int(max((h - w) / 2, 0)),
#         cv2.BORDER_CONSTANT,
#         value=[0],
#     )
#     cv2.imshow("source", img)

# sinogram = radon(img)
# cv2.imshow("sinogram", sinogram)
#
# # Odwrotny Radon
# tomograf = radon(sinogram, inverse=True, size=img.shape)

# cv2.imshow("tomograf", tomograf)
# cv2.waitKey()
