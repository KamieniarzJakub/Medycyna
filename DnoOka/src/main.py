import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from PIL import Image, UnidentifiedImageError, ImageFile
import gzip
import numpy as np
from gui.dno_oka_gui import view_sliders, view_dno_oka
import io
from lib.mse import calc_mse
from lib.przetwarzanie import preprocess_image, segment_vessels, postprocess_image
from lib.calc_diff import visualize_array_difference

supported_file_types = ["jpg", "jpeg", "png", "ppm", "gz", "webp", "avif", "gif"]

def read_img(fil: io.BytesIO | UploadedFile) -> Image.Image:
    try:
        return Image.open(fil).copy()
    except UnidentifiedImageError:
        if (isinstance(fil,UploadedFile)):
            raise Exception("Błędny typ pliku: ", fil.type, ", plik: ", fil.name, ", wspierane rozszerzenia: ", supported_file_types) 
        else:
            raise Exception("Błędny typ pliku; plik: ", fil.name, ", wspierane rozszerzenia: ", supported_file_types) 

def read_file(file: UploadedFile) -> Image.Image:
    if file.type == "application/gzip" or file.type == "application/x-gzip":
        with gzip.open(file,"rb") as f:
            return read_img(io.BytesIO(f.read()))
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
    image = read_file(file).convert("RGB")
    image_arr = np.asarray(image)

    st.image(image, caption="Oryginalny obraz", use_container_width=True, clamp=True)
    tab1, tab2, tab3 = st.tabs(["Wstępne przetwarzanie", "Segmentacja naczyń", "Postprocessing"])
    
    pre = preprocess_image(image_arr / 255.0)
    vessels = segment_vessels(pre)
    final = postprocess_image(vessels)
    diff: np.ndarray | None = None

    expected_image: Image.Image|None = None
    if expected_result is not None:
        expected_image = read_file(expected_result)
        expected_image_arr = np.asarray(expected_image)
        diff = visualize_array_difference(final,expected_image_arr)

    with tab1:
        st.image(pre, caption="Po wstępnym przetwarzaniu", use_container_width=True, clamp=True)

    with tab2:
        st.image(vessels, caption="Segmentacja naczyń (Frangi)", use_container_width=True, clamp=True)

    with tab3:
        st.image(final, caption="Po końcowym przetwarzaniu", use_container_width=True, clamp=True)

        if diff is not None and expected_image is not None:
            st.image(expected_image, "Obraz docelowy", clamp=True)

            img = Image.fromarray(diff, 'RGB')
            st.image(img, "Różnica względem obrazu docelowego", clamp=True)
            """
                - Na czerwono nadmiarowe wykrycia
                - Na czarno brak różnicy, czyli poprawne
                - Na niebiesko brakujące wykrycia
            """
            

            mse_result = (diff**2).mean()
            st.text("Błąd średniokwadratowy: " + str(mse_result))
