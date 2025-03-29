import cv2
import numpy as np
import pydicom
from PIL import Image
import streamlit as st
import os

from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.uid import CTImageStorage
from pydicom.uid import generate_uid
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

def convert_image_to_ubyte(img):
    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))

def save_as_dicom(file_name, img, patient_data={"PatientName":"", "PatientID":"", "ImageComments":""}):
    img_converted = convert_image_to_ubyte(img)
    
    # Populate required values for file meta information
    meta = Dataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  

    ds = FileDataset(None, {}, preamble=b"\0" * 128)
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    
    ds.PatientName = patient_data["PatientName"]
    ds.PatientID = patient_data["PatientID"]
    ds.ImageComments = patient_data["ImageComments"]
    

    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SamplesPerPixel = 1
    ds.HighBit = 7

    ds.ImagesInAcquisition = 1
    ds.InstanceNumber = 1

    ds.Rows, ds.Columns = img_converted.shape

    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    ds.PixelData = img_converted.tobytes()

    ds.save_as(file_name, write_like_original=False)

def read_dicom_file(file_name, tab):
    dcm_data = pydicom.dcmread(f"input/{file_name}")
    print(dcm_data)
    #Dane ogóle o pacjencie
    if not "PatientID" in dcm_data.dir():
        dcm_data.PatientID = ""
    dcm_data.PatientID = tab.text_input("PatientID", dcm_data.PatientID)
    if not "PatientName" in dcm_data.dir():
        dcm_data.PatientName = ""
    dcm_data.PatientName = tab.text_input("PatientName", dcm_data.PatientName)
    if not "PatientAge" in dcm_data.dir():
        dcm_data.PatientAge = ""
    dcm_data.PatientAge = tab.text_input("PatientAge", dcm_data.PatientAge)
    if not "PatientSex" in dcm_data.dir():
        dcm_data.PatientSex = ""
    dcm_data.PatientSex = tab.text_input("PatientSex", dcm_data.PatientSex)
    if not "PatientBirthDate" in dcm_data.dir():
        dcm_data.PatientBirthDate = ""
    dcm_data.PatientBirthDate = tab.text_input("PatientBirthDate", dcm_data.PatientBirthDate)
    #Data badania
    if not "StudyDate" in dcm_data.dir():
        dcm_data.StudyDate = ""
    dcm_data.StudyDate = tab.text_input("StudyDate", dcm_data.StudyDate)
    #Komentarze
    if not "ImageComments" in dcm_data.dir():
        dcm_data.ImageComments = ""
    dcm_data.ImageComments = tab.text_input("ImageComments", dcm_data.ImageComments)

    print(dcm_data.PatientName)

    #img = dcm_data.pixel_array
    #print(img)
    print("Zapisywanie")
    pydicom.dcmwrite(f"output/{file_name}", dcm_data)
    print("Zapisano!")

# DICOM
# a) Podstawowe informacje o pacjencie
# ID pacjenta -> PatientID
# imie i nazwisko -> PatientName
# wiek -> PatientAge
# płeć -> PatientSex
# data urodzenia -> PatientBirthDate

# b) Data badania -> StudyDate

# c) Komentarzy -> StudyComments lub ImageComments

# obraz -> pixel_array 


image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "dcm"])
if image is not None:
    file_details = {"FileName":image.name, "FileType":image.type}
    with open(f"input/{image.name}", "wb") as f:
        f.write(image.getbuffer())
    if not os.path.splitext(f"input/{image.name}")[1]==".dcm":
        save_as_dicom(f"input/{image.name.split('.')[0]}.dcm", cv2.imread(f"input/{image.name}", cv2.IMREAD_GRAYSCALE))
    tab1, tab2 = st.tabs(["Tomograf", "DICOM Data"])
    read_dicom_file(f"{image.name.split('.')[0]}.dcm", tab2)
    print()

