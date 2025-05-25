import cv2
from skimage import exposure, filters, morphology, measure
from scipy import ndimage
from PIL import ImageOps, Image
import numpy as np

def auto_contrast_bw(img: np.ndarray) -> np.ndarray:
    img_conv = (255 * img).astype(np.uint8)
    img_adj = ImageOps.autocontrast(
        Image.fromarray(
            img_conv,
            mode="L",
        ),
        0.1,
    )
    return np.asarray(img_adj) / 255

def preprocess_image(img):
    # 1. Ekstrakcja zielonego kanału jeśli RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, 1]

    # 2. Normalizacja do zakresu [0, 1]
    img = img / 255.0 if img.max() > 1 else img

    # 3. Histogram equalization (CLAHE)
    img_clahe = exposure.equalize_adapthist(img, clip_limit=0.05)

    # 4. Redukcja szumu (Gaussian Blur)
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)

    # 5. Wyostrzenie (Unsharp masking)
    # Wyostrzenie = obraz - rozmyty + wzmacnianie
    gaussian = cv2.GaussianBlur(img_blur, (0, 0), sigmaX=1)
    img_sharp = cv2.addWeighted(img_blur, 1.5, gaussian, -0.5, 0)

    # 6. Reskalowanie intensywności (opcjonalne)
    img_final = exposure.rescale_intensity(img_sharp, out_range=(0, 1)) # type: ignore

    return img_final


def segment_vessels(img):
    # Frangi vesselness filter (or Sobel for simplicity)
    vessel = filters.frangi(
        img,
        sigmas=np.arange(1,5), # type: ignore
        scale_step=1,
        alpha=0.3,
        beta=15,
        black_ridges=True
    )

    # Wzmocnienie kontrastu naczyń
    vessel = exposure.rescale_intensity(vessel, out_range=(0, 1)) # type: ignore
    return vessel

def postprocess_image(vessels):
    # Binary closing to connect broken vessel segments
    binary = vessels > 0.2
    cleaned = morphology.remove_small_objects(binary, min_size=64)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
    cleaned = auto_contrast_bw(cleaned)
    return cleaned


def divide_image_into_chunks(image: np.ndarray, chunk_size=(25, 25)):
    chunk_height, chunk_width = chunk_size

    padding_height = (chunk_height - image.shape[0] % chunk_height) % chunk_height
    padding_width = (chunk_width - image.shape[1] % chunk_width) % chunk_width

    if image.ndim == 3:  # Kolorowe
        padded_image = np.pad(image, ((0, padding_height), (0, padding_width), (0, 0)), mode='constant', constant_values=0)
    else:  # czarno białe
        padded_image = np.pad(image, ((0, padding_height), (0, padding_width)), mode='constant', constant_values=0)


    return np.array([padded_image[i:i + chunk_height, j:j + chunk_width] for j in range(0, padded_image.shape[1], chunk_width) for i in range(0,  padded_image.shape[0], chunk_height) ])

# def sliding_window(a, window, axis=-1):
#     shape = list(a.shape) + [window]
#     shape[axis] -= window - 1
#     if shape[axis] < 0:
#         raise ValueError("Array too small")
#     strides = a.strides + (a.strides[axis],)
#     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# def sliding_img_var(img, window):
#     if window <= 0:
#         raise ValueError("invalid window size")
#     buf = sliding_window(img, 2*window, 0)
#     buf = sliding_window(buf, 2*window, 1)

#     out = np.zeros(img.shape, dtype=np.float32)
#     np.var(buf[:-1,:-1], axis=(-1,-2), out=out[window:-window,window:-window])
#     return out


# def winVar(img, wlen):
#   wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
#     borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
#   return wsqrmean - wmean*wmean

def get_img_params(img):
    measure.moments_hu
    measure.moments_central
    # win_mean = ndimage.uniform_filter(img, (win_rows, win_cols))
    # win_sqr_mean = ndimage.uniform_filter(img**2, (win_rows, win_cols))
    # win_var = win_sqr_mean - win_mean**2