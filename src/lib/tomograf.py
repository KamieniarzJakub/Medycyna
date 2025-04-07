import numpy as np
from numba import jit


@jit
def plotLineLow(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0

    for x in range(x0, x1):
        yield (x, y)
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy


@jit
def plotLineHigh(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0

    for y in range(y0, y1):
        yield (x, y)
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx


@jit
def bresenham_line(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            for x, y in plotLineLow(x1, y1, x0, y0):
                yield (x, y)
        else:
            for x, y in plotLineLow(x0, y0, x1, y1):
                yield (x, y)
    else:
        if y0 > y1:
            for x, y in plotLineHigh(x1, y1, x0, y0):
                yield (x, y)
        else:
            for x, y in plotLineHigh(x0, y0, x1, y1):
                yield (x, y)


@jit
def radon(
    img,
    angle_step,
    num_emiters,
    detectors_angle=90,
    full_scan_range=360,
    run_convolve=False,
):
    w, h = img.shape
    out = np.zeros((full_scan_range // angle_step, num_emiters))

    for a, i, y, x in radon_step(
        angle_step, num_emiters, w, h, detectors_angle, full_scan_range
    ):
        out[a][i] += img[y][x]

    if run_convolve:
        kernel = convolve_kernel(n=21)
        for a in range(0, full_scan_range // angle_step):
            out[a] = np.convolve(out[a, :], kernel, mode="same")
    out /= out.max()
    return out


@jit
def inverse_radon(
    sinogram, size, angle_step, num_emiters, detectors_angle=90, full_scan_range=360
):
    h, w = size
    out = np.zeros(size)

    for a, i, y, x in radon_step(
        angle_step, num_emiters, w, h, detectors_angle, full_scan_range
    ):
        out[y][x] += sinogram[a][i]

    out /= out.max()
    return out


@jit
def radon_single_beam(angle, detectors_angle, shift, i, w, h):
    for x, y in bresenham_line(
        *radon_emiter_detector(angle, detectors_angle, shift, i, w, h)
    ):
        yield (i, y, x)


@jit
def radon_emiter_detector(angle, detectors_angle, shift, i, w, h):
    r = w / 2
    emiter = angle - detectors_angle / 2 + i * shift
    emiter_angle_rad = np.deg2rad(emiter)
    x_e = min(int(r - r * np.cos(emiter_angle_rad)), w - 1)
    y_e = min(int(r - r * np.sin(emiter_angle_rad)), h - 1)
    detector = angle + detectors_angle / 2 + 180 - i * shift
    detector_angle_rad = np.deg2rad(detector)
    x_d = min(int(r - r * np.cos(detector_angle_rad)), w - 1)
    y_d = min(int(r - r * np.sin(detector_angle_rad)), h - 1)

    return (x_e, y_e, x_d, y_d)


@jit
def radon_step(angle_step, n, w, h, detectors_angle, full_scan_range=360):
    shift = detectors_angle / (n - 1)
    for a, angle in enumerate(range(0, full_scan_range, angle_step)):
        for i in range(n):
            for b in radon_single_beam(angle, detectors_angle, shift, i, w, h):
                yield (a, *b)


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
