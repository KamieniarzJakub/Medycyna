import pydicom
import streamlit as st
from PIL import Image
import gzip
import numpy as np
from lib import img_processing
from lib.dicomloader import create_DICOM
from gui.tomograf_gui import view_sliders, view_tomograf
from gui.dicom_gui import dicom_file_gui
import io
from pydicom.filebase import DicomFileLike
from lib.mse import calc_mse
import os
from lib.przetwarzanie import preprocess_image, segment_vessels, postprocess_image

DICOM_MIME = "application/dicom"


st.title("Wykrywanie naczyń dna siatkówki oka")
file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "dcm", "ppm", "gz"], accept_multiple_files=False
)
if file is not None:
    file_details = {"FileName": file.name, "FileType": file.type}

    image: np.ndarray
    if file.type == "application/gzip" or file.type == "application/x-gzip":
        with gzip.open(file) as f:
            image = np.asarray(Image.open(f))
    elif file.type == "application/ppm":
        image = np.asarray(Image.open(file))
    elif file.type=="application/octet-stream":
        image = np.asarray(Image.open(file))
    else:
        raise Exception("Błędy typ pliku") 

    st.image(image, caption="Oryginalny obraz", use_column_width=True, clamp=True)
    tab1, tab2, tab3 = st.tabs(["Wstępne przetwarzanie", "Segmentacja naczyń", "Postprocessing"])
    
    with tab1:
        pre = preprocess_image(image / 255.0)
        st.image(pre, caption="Po wstępnym przetwarzaniu", use_column_width=True, clamp=True)

    with tab2:
        vessels = segment_vessels(pre)
        st.image(vessels, caption="Segmentacja naczyń (Frangi)", use_column_width=True, clamp=True)

    with tab3:
        final = postprocess_image(vessels)
        st.image(final, caption="Po końcowym przetwarzaniu", use_column_width=True, clamp=True)
    