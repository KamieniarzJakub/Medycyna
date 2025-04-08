import pydicom
import streamlit as st
from PIL import Image
import numpy as np
import math
from lib.dicomloader import create_DICOM
from gui.tomograf_gui import view_sliders, view_tomograf
from gui.dicom_gui import dicom_file_gui
import io
from pydicom.filebase import DicomFileLike
from lib.mse import calc_mse

DICOM_MIME = "application/dicom"



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
    if file.type == DICOM_MIME or file.type=="application/octet-stream": #u mnie jest inny typ pliku - nie wiem dlaczego, nie pytam
        dcm_data = pydicom.dcmread(file)
        image = dcm_data.pixel_array
    else:
        print(file.type)
        image = np.asarray(Image.open(file).convert("L"))
        dcm_data = create_DICOM(image)

    if image.shape[0] != image.shape[1]:
        image = add_borders_to_rectangle(image)

    params = tuple()
    with st.sidebar:
        tab1, tab2 = st.tabs(["Parametry tomografu", "Dane DICOM"])
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

        mse_result = calc_mse(image, reconstructed)
        st.text("Błąd średniokwadratowy: " + str(mse_result))
