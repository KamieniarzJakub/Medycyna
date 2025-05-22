import streamlit as st
from PIL import Image, UnidentifiedImageError, ImageFile
import gzip
import numpy as np
from lib import img_processing
from gui.dno_oka_gui import view_sliders, view_dno_oka
import io
from lib.mse import calc_mse
from lib.przetwarzanie import preprocess_image, segment_vessels, postprocess_image
from lib.calc_diff import visualize_array_difference

supported_file_types = ["jpg", "jpeg", "png", "ppm", "gz", "webp", "avif", "gif"]

def read_img(fil: io.BytesIO) -> ImageFile:
    try:
        return Image.open(fil).copy()
    except UnidentifiedImageError:
        raise Exception("Błędny typ pliku: ", fil.type, ", plik: ", fil.name, ", wspierane rozszerzenia: ", supported_file_types) 

def read_file(file: io.BytesIO) -> ImageFile:
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
    image_arr = np.asarray(image)

    st.image(image, caption="Oryginalny obraz", use_column_width=True, clamp=True)
    tab1, tab2, tab3 = st.tabs(["Wstępne przetwarzanie", "Segmentacja naczyń", "Postprocessing"])
    
    pre = preprocess_image(image_arr / 255.0)
    vessels = segment_vessels(pre)
    final = postprocess_image(vessels)

    expected_image: np.ndarray
    if expected_result is not None:
        expected_image_arr = np.asarray(read_file(expected_result).convert("L"))
        diff = visualize_array_difference(image_arr,expected_image_arr)

    with tab1:
        st.image(pre, caption="Po wstępnym przetwarzaniu", use_column_width=True, clamp=True)

    with tab2:

        st.image(vessels, caption="Segmentacja naczyń (Frangi)", use_column_width=True, clamp=True)

    with tab3:
        st.image(final, caption="Po końcowym przetwarzaniu", use_column_width=True, clamp=True)

        if expected_result is not None:
            st.image(diff, "Różnica względem obrazu docelowego", clamp=True)

            mse_result = (np.asarray(diff)**2).mean()
            st.text("Błąd średniokwadratowy: " + str(mse_result))
