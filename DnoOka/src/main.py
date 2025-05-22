import streamlit as st
from PIL import Image, UnidentifiedImageError
import gzip
import numpy as np
from lib import img_processing
from gui.tomograf_gui import view_sliders, view_dno_oka
import io
from lib.mse import calc_mse

supported_file_types = ["jpg", "jpeg", "png", "ppm", "gz", "webp", "avif", "gif"]

def read_img(fil):
    try:
        return np.asarray(Image.open(fil))
    except UnidentifiedImageError:
        raise Exception("Błędny typ pliku: ", fil.type, ", plik: ", fil.name, ", wspierane rozszerzenia: ", supported_file_types) 

st.title("Wykrywanie naczyń dna siatkówki oka")
file = st.file_uploader(
    "Skan siatkówki oka do analizy", type=supported_file_types, accept_multiple_files=False
)

expected_result = st.file_uploader(
    "[Opcjonalne] docelowy obraz naczyń", type=supported_file_types, accept_multiple_files=False
)

if file is not None:
    file_details = {"FileName": file.name, "FileType": file.type}


    image: np.ndarray
    if file.type == "application/gzip" or file.type == "application/x-gzip":
        with gzip.open(file) as f:
            image = read_img(f)
    else:
        image = read_img(file)

    params = tuple()
    with st.sidebar:
        tab1, = st.tabs(["Parametry przetwarzania obrazu"])
    params = view_sliders(tab1)

    if len(params) > 0:
        reconstructed = view_dno_oka(st, image, *params)

        mse_result = calc_mse(image / 255, reconstructed)
        st.text("Błąd średniokwadratowy: " + str(mse_result))
