import numpy as np
from PIL import Image
from numba import jit

@jit
def visualize_array_difference(array1: np.ndarray, array2: np.ndarray) -> Image.Image | None:
    """
    Visualizes the element-wise difference between two NumPy arrays as an image.

    The image uses a color scheme to represent the difference:
    - Negative values: Shades of red (darker red for larger absolute negative values).
    - Zero values: Black.
    - Positive values: Shades of blue (darker blue for larger positive values).

    Args:
        array1 (numpy.ndarray): The first NumPy array.
        array2 (numpy.ndarray): The second NumPy array.

    Returns:
        PIL.Image.Image: A Pillow Image object representing the difference.
                         Returns None if the input arrays have different shapes.
    """
    # Ensure arrays have the same shape
    if array1.shape != array2.shape:
        print(f"Error: Input arrays must have the same shape. Got {array1.shape} and {array2.shape}")
        return None

    # Calculate the element-wise difference
    diff_array = array2 - array1

    # Get dimensions of the difference array (which are the image dimensions)
    height, width = diff_array.shape

    # Create a new RGB image with a black background
    # 'RGB' mode means 3 channels (Red, Green, Blue)
    img = Image.new('RGB', (width, height), color='black')

    # Get a pixel access object to set pixel colors directly
    pixels = img.load()

    # Determine the minimum and maximum difference values for scaling color intensity.
    # These are used to normalize the differences to a 0-255 range for color channels.
    min_diff = np.min(diff_array)
    max_diff = np.max(diff_array)


    def get_color(value):
        red_intensity = 0
        blue_intensity = 0

        if value < 0:
            # For negative values, calculate red intensity.
            # The intensity is proportional to how negative the value is,
            # relative to the most negative value (min_diff).
            # abs(value) / abs(min_diff) scales from 0 to 1.
            # We only scale if min_diff is actually negative to avoid division by zero
            # and ensure correct behavior if all differences are non-negative.
            if min_diff < 0:
                red_intensity = int(255 * (abs(value) / abs(min_diff)))
            # If min_diff is 0 or positive, it means there are no negative values,
            # so red_intensity remains 0.
        elif value > 0:
            # For positive values, calculate blue intensity.
            # The intensity is proportional to how positive the value is,
            # relative to the most positive value (max_diff).
            # value / max_diff scales from 0 to 1.
            # We only scale if max_diff is actually positive to avoid division by zero
            # and ensure correct behavior if all differences are non-positive.
            if max_diff > 0:
                blue_intensity = int(255 * (value / max_diff))
            # If max_diff is 0 or negative, it means there are no positive values,
            # so blue_intensity remains 0.
        # If value == 0, both red_intensity and blue_intensity remain 0,
        # resulting in a black pixel (0, 0, 0).
        return (red_intensity, 0, blue_intensity)

    vfunc = np.vectorize(get_color)
    pixels = vfunc(diff_array)

    return img