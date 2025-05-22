import streamlit as st
from PIL import Image, UnidentifiedImageError
import gzip
import numpy as np
from lib import img_processing
from gui.tomograf_gui import view_sliders, view_dno_oka
import io
from lib.mse import calc_mse
from lib.calc_diff import visualize_array_difference

supported_file_types = ["jpg", "jpeg", "png", "ppm", "gz", "webp", "avif", "gif"]

def read_img(fil: io.BytesIO) -> np.ndarray:
    try:
        return np.asarray(Image.open(fil))
    except UnidentifiedImageError:
        raise Exception("Błędny typ pliku: ", fil.type, ", plik: ", fil.name, ", wspierane rozszerzenia: ", supported_file_types) 

def read_file(file):
    if file.type == "application/gzip" or file.type == "application/x-gzip":
        with gzip.open(file) as f:
            return read_img(f)
    else:
        return read_img(file)

st.title("Wykrywanie naczyń dna siatkówki oka")
file = st.file_uploader(
    "Skan siatkówki oka do analizy", type=supported_file_types, accept_multiple_files=False
)

expected_result = st.file_uploader(
    "[Opcjonalne] docelowy obraz naczyń", type=supported_file_types, accept_multiple_files=False
)

if file is not None:
    image = read_file(file)

    expected_image: np.ndarray
    if expected_result is not None:
        expected_image = read_file(expected_result)

    params = tuple()
    with st.sidebar:
        tab1, = st.tabs(["Parametry przetwarzania obrazu"])
    params = view_sliders(tab1)

    if len(params) > 0:
        reconstructed = view_dno_oka(st, image, *params)

        if expected_result is not None:
            diff = visualize_array_difference(image,expected_image)
            st.image(diff, "Różnica względem obrazu docelowego", clamp=True)

        mse_result = calc_mse(image / 255, reconstructed)
        st.text("Błąd średniokwadratowy: " + str(mse_result))
