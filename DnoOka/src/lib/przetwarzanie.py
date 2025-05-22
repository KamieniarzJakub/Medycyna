import cv2
from skimage import exposure, filters, morphology

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
    img_final = exposure.rescale_intensity(img_sharp, out_range=(0, 1))

    return img_final


def segment_vessels(img):
    # Frangi vesselness filter (or Sobel for simplicity)
    vessel = filters.frangi(
        img,
        scale_range=(1, 5),
        scale_step=1,
        alpha=0.3,
        beta=15,
        black_ridges=True
    )

    # Wzmocnienie kontrastu naczyń
    vessel = exposure.rescale_intensity(vessel, out_range=(0, 1))
    return vessel

def postprocess_image(vessels):
    # Binary closing to connect broken vessel segments
    binary = vessels > 0.2
    cleaned = morphology.remove_small_objects(binary, min_size=64)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
    return cleaned