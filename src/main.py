import pydicom
import streamlit as st
from PIL import Image
import numpy as np
import uuid
from lib.dicomloader import create_DICOM
from gui.tomograf_gui import view_tomograf
from gui.dicom_gui import dicom_file_gui

st.title("Tomograf")
file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "dcm"], accept_multiple_files=False
)
if file is not None:
    file_details = {"FileName": file.name, "FileType": file.type}

    dcm_data: pydicom.Dataset
    image: np.ndarray
    if file.type == "dcm":
        dcm_data = pydicom.FileDataset(file, pydicom.Dataset())
        image = dcm_data.get("ImageData")
    else:
        image = np.asarray(Image.open(file))
        dcm_data = create_DICOM(image)

    tab1, tab2 = st.tabs(["Tomograf", "DICOM Data"])

    view_tomograf(tab1, image)
    dicom_file_gui(tab2, dcm_data)
