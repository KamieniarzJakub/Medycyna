import numpy as np
from numba import jit
import cv2


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
def radon_init(angle_step, num_emiters, full_scan_range=360):
    scans_angle_labels = np.linspace(
        0, full_scan_range, np.ceil(full_scan_range / angle_step)
    )
    out = np.zeros((len(scans_angle_labels), num_emiters))
    return scans_angle_labels, out


# @jit
def radon(img, angle_step, num_emiters, full_scan_range=360):
    w, h = img.shape
    # scans_angle_labels, out = radon_init(angle_step, num_emiters, full_scan_range)
    out = np.zeros((full_scan_range // angle_step, num_emiters))
    # img2 = np.zeros(img.shape)
    for a, i, y, x in radon_step(angle_step, num_emiters, w, h, 90, out):
        # img2[y][x] = 255
        # cv2.imshow("source 2", img2)
        # cv2.waitKey(1)
        out[a][i] += img[y][x]

    out /= out.max()
    return out


# @jit
def inverse_radon(sinogram, size, angle_step, num_emiters):
    h, w = size
    out = np.zeros(size)

    for a, i, y, x in radon_step(angle_step, num_emiters, w, h, 90, out):
        out[y][x] += sinogram[a][i]

    out /= out.max()
    return out


def radon_single_beam(angle, detectors_angle, shift, i, r):
    emiter = angle - detectors_angle / 2 + i * shift
    emiter_angle_rad = np.deg2rad(emiter)
    x_e = min(int(r - r * np.cos(emiter_angle_rad)), w - 1)
    y_e = min(int(r - r * np.sin(emiter_angle_rad)), h - 1)
    detector = angle + detectors_angle / 2 + 180 - i * shift
    detector_angle_rad = np.deg2rad(detector)
    x_d = min(int(r - r * np.cos(detector_angle_rad)), w - 1)
    y_d = min(int(r - r * np.sin(detector_angle_rad)), h - 1)

    line = []
    for x, y in bresenham_line(x_e, y_e, x_d, y_d):
        line.append((i, y, x))
    return line


def radon_emiter_detector(angle, detectors_angle, shift, i, r):
    emiter = angle - detectors_angle / 2 + i * shift
    emiter_angle_rad = np.deg2rad(emiter)
    x_e = min(int(r - r * np.cos(emiter_angle_rad)), w - 1)
    y_e = min(int(r - r * np.sin(emiter_angle_rad)), h - 1)
    detector = angle + detectors_angle / 2 + 180 - i * shift
    detector_angle_rad = np.deg2rad(detector)
    x_d = min(int(r - r * np.cos(detector_angle_rad)), w - 1)
    y_d = min(int(r - r * np.sin(detector_angle_rad)), h - 1)

    return (x_e, y_e, x_d, y_d)


# @jit
def radon_step(angle_step, n, w, h, detectors_angle, out):
    r = w / 2
    shift = detectors_angle / (n - 1)
    for a, angle in enumerate(range(0, 360, angle_step)):
        for i in range(n):
            for b in radon_single_beam(angle, detectors_angle, shift, i, r):
                yield (a, *b)
        # cv2.imshow("sinogram", out)
        # cv2.waitKey(1)


def radom_full(
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


@jit
def convolve_kernel(n=21):
    kernel = np.zeros((n,))
    mid = n // 2
    c = -4 / np.pi**2
    for i in range(1, mid, 2):
        val = c / (i) ** 2
        kernel[mid + i] = val
        kernel[mid - i] = val
    kernel[mid] = 1
    return kernel


if __name__ == "__main__":
    img = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
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
    cv2.waitKey(1)

    sinogram = radon(img, 4, 180)
    # sinogram = radom_full(img, img.shape, 4, 180, 30)
    cv2.imshow("sinogram", sinogram)
    cv2.waitKey(1)

    # Odwrotny Radon
    # tomograf = radom_full(sinogram, img.shape, 4, 180, 30, inverse=True)
    tomograf = inverse_radon(sinogram, img.shape, 4, 180)

    cv2.imshow("tomograf", tomograf)
    cv2.waitKey()
    cv2.destroyAllWindows()
