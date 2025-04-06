import numpy as np
from numba import jit


@jit
def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        yield (x0, y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


@jit
def radon(img, angle_step, num_emiters):
    h, w = img.shape
    out = np.zeros((360 // angle_step, num_emiters))

    for a, i, y, x in radon_step(angle_step, num_emiters, w, h):
        out[a][i] += img[y][x]

    out /= out.max()
    return out


@jit
def inverse_radon(sinogram, size, angle_step, num_emiters):
    h, w = size
    out = np.zeros(size)

    for a, i, y, x in radon_step(angle_step, num_emiters, w, h):
        out[y][x] += sinogram[a][i]

    out /= out.max()
    return out


@jit
def radon_step(angle_step, n, w, h):
    r = w / 2
    detectors_angle = 90
    for a, angle in enumerate(range(0, 360, angle_step)):
        for i in range(n):
            emiter = angle - i * (detectors_angle / n - 1)
            emiter_angle_rad = np.deg2rad(emiter)
            x_e = min(int(r - r * np.cos(emiter_angle_rad)), w - 1)
            y_e = min(int(r - r * np.sin(emiter_angle_rad)), h - 1)
            detector = angle + 180 + i * (detectors_angle / n - 1)
            detector_angle_rad = np.deg2rad(detector)
            # print(emiter, detector)
            x_d = min(int(r - r * np.cos(detector_angle_rad)), w - 1)
            y_d = min(int(r - r * np.sin(detector_angle_rad)), h - 1)

            for x, y in bresenham_line(x_e, y_e, x_d, y_d):
                yield (a, i, y, x)


@jit
def convolve_kernel():
    n = 22
    kernel = np.zeros((n,))
    for i in range(1, n, 2):
        val = -4 / np.pi**2 / i**2
        kernel[i] = val
        kernel[-i] = val
    kernel[0] = 1
    return kernel


if __name__ == "__main__":
    import cv2

    img = cv2.imread("img/Shepp_logan.jpg", cv2.IMREAD_GRAYSCALE)
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

    sinogram = radon(img, 2, 500)
    cv2.imshow("sinogram", sinogram)

    # Odwrotny Radon
    tomograf = inverse_radon(sinogram, img.shape, 2, 500)

    cv2.imshow("tomograf", tomograf)
    cv2.waitKey()
