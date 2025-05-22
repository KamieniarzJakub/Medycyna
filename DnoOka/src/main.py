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

DICOM_MIME = "application/dicom"


st.title("Wykrywanie naczyń dna siatkówki oka")
file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "dcm", "ppm", "gz"], accept_multiple_files=False
)
if file is not None:
    file_details = {"FileName": file.name, "FileType": file.type}

    dcm_data: pydicom.Dataset
    image: np.ndarray
    if file.type == DICOM_MIME:
        dcm_data = pydicom.dcmread(file)
        image = dcm_data.pixel_array
    elif file.type == "application/gzip" or file.type == "application/x-gzip":
        with gzip.open(file) as f:
            image = np.asarray(Image.open(f).convert("L"))
        dcm_data = create_DICOM(image)
    elif file.type == "application/ppm":
        image = np.asarray(Image.open(file).convert("L"))
        dcm_data = create_DICOM(image)
    elif file.type=="application/octet-stream":
        filename = file.name.lower()
        extension = os.path.splitext(filename)[1]

        if extension in [".dcm"]:
            dcm_data = pydicom.dcmread(file)
            image = dcm_data.pixel_array
        else:
            image = np.asarray(Image.open(file).convert("L"))
            dcm_data = create_DICOM(image)
    else:
        raise Exception("Błędny typ pliku: " + file.type + ", rozszerzenie: " + os.path.splitext(filename)[1:]) 

    if image.shape[0] != image.shape[1]:
        image = img_processing.add_borders_to_rectangle(image)

    params = tuple()
    with st.sidebar:
        tab1, tab2 = st.tabs(["Parametry przetwarzania obrazu", "Dane DICOM"])
        dfg = dicom_file_gui(tab2, dcm_data)
    params = view_sliders(tab1)

    if len(params) > 0:
        reconstructed = view_tomograf(st, image, *params)

        reconstructed_scaled = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)

        dcm_data.Rows, dcm_data.Columns = reconstructed_scaled.shape
        dcm_data.PixelData = reconstructed_scaled.tobytes()
        dcm_data.BitsAllocated = 8
        dcm_data.BitsStored = 8
        dcm_data.HighBit = 7
        dcm_data.SamplesPerPixel = 1
        dcm_data.PhotometricInterpretation = "MONOCHROME2"

        with io.BytesIO() as buf:
            mem_dataset = DicomFileLike(buf)
            for k, v in dfg.items():
                setattr(dcm_data, k, v)
            pydicom.dcmwrite(mem_dataset, dcm_data, write_like_original=False)
            mem_dataset.seek(0)
            st.download_button(
                label="Download DICOM",
                data=mem_dataset.read(),
                file_name="result.dcm",
                mime=DICOM_MIME,
                icon=":material/download:",
            )

        mse_result = calc_mse(image / 255, reconstructed)
        st.text("Błąd średniokwadratowy: " + str(mse_result))
