import numpy as np
from skimage.measure import moments_central, moments_hu

def extract_features_from_patch(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 3:
        patch = patch[:, :, 1]  # Użyj zielonego kanału do ekstrakcji cech
    patch = patch.astype(np.float32)

    # Statystyki pierwszego rzędu
    mean_val = np.mean(patch)
    std_val = np.std(patch)

    # Momenty centralne i momenty Hu
    center = (patch.shape[0] // 2, patch.shape[1] // 2)
    m_central = moments_central(patch, center=center, order=3)
    hu = moments_hu(m_central)

    # Zwracanie połączonych cech
    return np.hstack([mean_val, std_val, hu])

