import cv2
import numpy as np
import streamlit as st
import os



# def convert_image_to_ubyte(img):
#     return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "dcm"])
if image is not None:
    file_details = {"FileName": image.name, "FileType": image.type}
    extension = os.path.splitext(f"input/{os.name}")[1]

    with open(f"input/{image.name}", "wb") as f:
        f.write(image.getbuffer())
    if not  == ".dcm":
        save_as_dicom(
            f"input/{os.path.basename(image.name)}.dcm",
            image,
        )
    tab1, tab2 = st.tabs(["Tomograf", "DICOM Data"])
    read_dicom_file(f"{image.name.split('.')[0]}.dcm", tab2)

    # img = cv2.imread("input/" + image.name, cv2.IMREAD_GRAYSCALE)
    nparr = np.frombuffer(image.getbuffer(), dtype=np.ubyte)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    w, h = img.shape
    # img = cv2.copyMakeBorder(
    #     img,
    #     int(max((w - h) / 1.4, 0)),
    #     int(max((w - h) / 1.4, 0)),
    #     int(max((h - w) / 1.4, 0)),
    #     int(max((h - w) / 1.4, 0)),
    #     cv2.BORDER_CONSTANT,
    #     value=[0],
    # )
    s.view_tomograf(tab1, img)
