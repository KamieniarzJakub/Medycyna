import pydicom
import streamlit as st
from PIL import Image
import numpy as np
import math
import uuid
from lib.dicomloader import create_DICOM
from gui.tomograf_gui import view_sliders, view_tomograf
from gui.dicom_gui import dicom_file_gui


def add_borders_to_rectangle(img):
    h, w = img.shape
    m = max(w, h)
    pw = (
        (math.floor((m - h) / 2), math.ceil((m - h) / 2)),
        (math.floor((m - w) / 2), math.ceil((m - w) / 2)),
    )
    return np.pad(
        img,
        pw,
        "constant",
        constant_values=(0),
    )


st.title("Tomograf")
file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "dcm"], accept_multiple_files=False
)
if file is not None:
    file_details = {"FileName": file.name, "FileType": file.type}

    dcm_data: pydicom.Dataset
    image: np.ndarray
    if file.type == "application/dicom":
        dcm_data = pydicom.FileDataset(file, pydicom.Dataset())
        print(type(dcm_data.get("ImageData")))
        image = dcm_data.get("ImageData")
    else:
        image_pil = Image.open(file).convert("L")
        image = np.asarray(image_pil)
        dcm_data = create_DICOM(image)

    if image.shape[0] != image.shape[1]:
        image = add_borders_to_rectangle(image)

    params = tuple()
    with st.sidebar:
        tab1, tab2 = st.tabs(["Parametry tomografu", "Dane DICOM"])
        dicom_file_gui(tab2, dcm_data)
        params = view_sliders(tab1)

    if len(params) > 0:
        view_tomograf(st, image, *params)
