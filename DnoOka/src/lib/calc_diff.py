import numpy as np
from PIL import Image

def get_color(value,min_diff,max_diff):
    red_intensity = 0
    blue_intensity = 0

    if value < 0:
        if min_diff < 0:
            red_intensity = int(255 * (abs(value) / abs(min_diff)))
    elif value > 0:
        if max_diff > 0:
            blue_intensity = int(255 * (value / max_diff))
    return np.array([red_intensity, 0, blue_intensity], dtype=np.uint8)

def visualize_array_difference(array1: np.ndarray, array2: np.ndarray) -> np.ndarray | None:
    """
    Visualizes the element-wise difference between two arrays as an image.

    The image uses a color scheme to represent the difference:
    - Negative values: Shades of red (more intense red for smaller negative values, redundant values).
    - Zero values: Black. (no difference)
    - Positive values: Shades of blue (more intense blue for larger positive values, missing values).
    """
    if array1.shape != array2.shape:
        print(f"Error: Input arrays must have the same shape. Got {array1.shape} and {array2.shape}")
        return None

    diff_array: np.ndarray[np.int16] = (array2.astype(np.int16) - array1.astype(np.int16))
    height, width = diff_array.shape

    min_diff = np.min(diff_array)
    max_diff = np.max(diff_array)

    vfunc = np.vectorize(get_color, signature='(),(),()->(n)')
    pixels = vfunc(diff_array,min_diff,max_diff)

    return pixels