import numpy as np
from numba import jit

@jit(cache=True)
def plotLineLow(x0, y0, x1, y1):
    points = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0

    for x in range(x0, x1 + 1):
        points.append((x, y))
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy
    return points

@jit(cache=True)
def plotLineHigh(x0, y0, x1, y1):
    points = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0

    for y in range(y0, y1 + 1):
        points.append((x, y))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx
    return points

@jit(cache=True)
def bresenham_line(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plotLineLow(x1, y1, x0, y0)[::-1]
        else:
            return plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return plotLineHigh(x1, y1, x0, y0)[::-1]
        else:
            return plotLineHigh(x0, y0, x1, y1)

@jit(cache=True)
def radon(img, angle_step, num_emiters, detectors_angle=90, full_scan_range=360, run_convolve=False):
    w, h = img.shape
    angle_values = np.arange(0, full_scan_range, angle_step)
    out = np.zeros((len(angle_values), num_emiters))
    shift = detectors_angle / (num_emiters - 1)

    for a, angle in enumerate(angle_values):
        for i in range(num_emiters):
            beam = radon_single_beam(angle, detectors_angle, shift, i, w, h)
            for y, x in beam:
                out[a, i] += img[y, x] / len(beam)

    if run_convolve:
        kernel = convolve_kernel(n=21)
        for a in range(len(angle_values)):
            out[a] = np.convolve(out[a], kernel, mode="same")

    out /= out.max()
    return out

@jit(cache=True)
def inverse_radon(sinogram, size, angle_step, num_emiters, detectors_angle=90, full_scan_range=360):
    h, w = size
    out = np.zeros(size)
    angle_values = np.arange(0, full_scan_range, angle_step)
    shift = detectors_angle / (num_emiters - 1)

    for a, angle in enumerate(angle_values):
        for i in range(num_emiters):
            beam = radon_single_beam(angle, detectors_angle, shift, i, w, h)
            for y, x in beam:
                out[y, x] += sinogram[a, i] / len(beam)

    out /= out.max()
    return out

@jit(cache=True)
def radon_single_beam(angle, detectors_angle, shift, i, w, h):
    points = []
    for x, y in bresenham_line(
        *radon_emiter_detector(angle, detectors_angle, shift, i, w, h)
    ):
        if 0 <= x < w and 0 <= y < h:
            points.append((y, x))
    return points

@jit(cache=True)
def radon_emiter_detector(angle, detectors_angle, shift, i, w, h):
    r = w / 2
    emiter = angle - detectors_angle / 2 + i * shift
    emiter_angle_rad = np.deg2rad(emiter)
    x_e = min(int(r * (1 - np.cos(emiter_angle_rad))), w - 1)
    y_e = min(int(r * (1 - np.sin(emiter_angle_rad))), h - 1)
    detector = angle + detectors_angle / 2 + 180 - i * shift
    detector_angle_rad = np.deg2rad(detector)
    x_d = min(int(r * (1 - np.cos(detector_angle_rad))), w - 1)
    y_d = min(int(r * (1 - np.sin(detector_angle_rad))), h - 1)

    return (x_e, y_e, x_d, y_d)

@jit(cache=True)
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
