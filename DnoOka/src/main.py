import streamlit as st
from PIL import Image
import gzip
import numpy as np
from lib import img_processing
from gui.tomograf_gui import view_sliders, view_dno_oka
import io
from lib.mse import calc_mse

st.title("Wykrywanie naczyń dna siatkówki oka")
file = st.file_uploader(
    "Skan siatkówki oka do analizy", type=["jpg", "jpeg", "png", "dcm", "ppm", "gz"], accept_multiple_files=False
)

expected_result = st.file_uploader(
    "[Opcjonalne] docelowy obraz naczyń", type=["jpg", "jpeg", "png", "dcm", "ppm", "gz"], accept_multiple_files=False
)

if file is not None:
    file_details = {"FileName": file.name, "FileType": file.type}

    image: np.ndarray
    if file.type == "application/gzip" or file.type == "application/x-gzip":
        with gzip.open(file) as f:
            image = np.asarray(Image.open(f).convert("L"))
    elif file.type == "application/ppm":
        image = np.asarray(Image.open(file).convert("L"))
    elif file.type=="application/octet-stream":
        image = np.asarray(Image.open(file).convert("L"))
    else:
        raise Exception("Błędny typ pliku: " + file.type + ", plik: " + file.name) 

    params = tuple()
    with st.sidebar:
        tab1, = st.tabs(["Parametry przetwarzania obrazu"])
    params = view_sliders(tab1)

    if len(params) > 0:
        reconstructed = view_dno_oka(st, image, *params)

        mse_result = calc_mse(image / 255, reconstructed)
        st.text("Błąd średniokwadratowy: " + str(mse_result))
