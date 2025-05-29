import cv2
from skimage import exposure, filters, morphology, measure
from skimage.filters import unsharp_mask
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

def preprocess_image(image_arr):


    img_green= image_arr[:, :, 1]

    # Ensure the input is in the range [0, 255] and of type uint8
    if img_green.dtype != np.uint8:
        img_green = (img_green * 255).astype(np.uint8)

    # Convert to grayscale
    # img_gray = cv2.cvtColor(img_green, cv2.COLOR_RGB2GRAY)

    # Clip the pixel values
    img_gray_clipped = np.clip(img_green, 10, 245).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray_norm = clahe.apply(img_gray_clipped)

    return img_gray_norm

    # # 2. Normalizacja do zakresu [0, 1]
    # img = img / 255.0 if img.max() > 1 else img

    # # 3. Histogram equalization (CLAHE)
    # img_clahe = exposure.equalize_adapthist(img, clip_limit=0.05)


    # 4. Redukcja szumu (Gaussian Blur)
    # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    # image = clahe.apply(img_blur)

    # # 5. Wyostrzenie (Unsharp masking)
    # # Wyostrzenie = obraz - rozmyty + wzmacnianie
    # gaussian = cv2.GaussianBlur(img_blur, (0, 0), sigmaX=1)
    # img_sharp = cv2.addWeighted(img_blur, 1.5, gaussian, -0.5, 0)

    # # 6. Reskalowanie intensywności (opcjonalne)
    # img_final = exposure.rescale_intensity(img_sharp, out_range=(0, 1)) # type: ignore

    # return img_gray_norm


def segment_vessels(img):
    # Wstępne przetwarzanie
    # img = cv2.GaussianBlur(img, (3,3), 0)
    
    # Frangi
    vessel = filters.frangi(
        img,
        sigmas=np.arange(1, 5),
        alpha=0.5,
        beta=0.5,
        black_ridges=True
    )

    # Kontrast + wyostrzanie
    vessel = exposure.rescale_intensity(vessel, out_range=(0, 1))
    vessel = exposure.equalize_adapthist(vessel, clip_limit=0.03)
    vessel = unsharp_mask(vessel, radius=1.0, amount=2.0)

    # Progowanie adaptacyjne
    threshold = np.percentile(vessel, 99)
    vessel_bin = (vessel > threshold).astype(np.uint8) * 255

    return vessel_bin

# def segment_vessels(img):
#     # Frangi vesselness filter (or Sobel for simplicity)
#     vessel = filters.frangi(
#         img,
#         sigmas=np.arange(1,5), # type: ignore
#         scale_step=1,
#         alpha=0.3,
#         beta=15,
#         black_ridges=True
#     )

#     # Wzmocnienie kontrastu naczyń
#     vessel = exposure.rescale_intensity(vessel, out_range=(0, 1)) # type: ignore
#     return vessel

def postprocess_image(vessels):
    # Binary closing to connect broken vessel segments
    binary = vessels > 0.2
    cleaned = morphology.remove_small_objects(binary, min_size=100)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(3))
    # cleaned = auto_contrast_bw(cleaned)
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

def get_img_params(img):
    measure.moments_hu
    measure.moments_central
