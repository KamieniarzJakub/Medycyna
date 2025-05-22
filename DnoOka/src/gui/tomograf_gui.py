from lib import tomograf
import numpy as np
from PIL import Image, ImageOps


def view_sliders(st):
    auto_contrast = st.checkbox("Auto kontrast")

    return (
        auto_contrast,
    )


def view_dno_oka(
    st,
    image,
    auto_contrast,
):
    st.image(image, "Obraz źródłowy", clamp=True)

    if auto_contrast:
        reconstructed_conv = (255 * reconstructed).astype(np.uint8)
        reconstructed_adj = ImageOps.autocontrast(
            Image.fromarray(
                reconstructed_conv,
                mode="L",
            ),
            0.1,
        )
        reconstructed = np.asarray(reconstructed_adj) / 255

    reconstructed = image
    # st.image(reconstructed, "Obraz po odtworzeniu")

    return reconstructed
