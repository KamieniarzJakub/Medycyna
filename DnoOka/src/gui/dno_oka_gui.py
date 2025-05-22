import numpy as np
from PIL import Image, ImageOps
from lib.img_processing import auto_contrast_bw


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

    # if auto_contrast:
        # auto_contrast_bw(image)

    reconstructed = image
    # st.image(reconstructed, "Obraz po odtworzeniu")

    return reconstructed
