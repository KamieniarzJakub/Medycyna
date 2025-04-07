import pydicom
import numpy as np
from pydicom.dataset import FileDataset, validate_file_meta
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.uid import CTImageStorage
from pydicom.uid import generate_uid


def save_as_dicom(
    file_name,
    img,
    patient_data={"PatientName": "", "PatientID": "", "ImageComments": ""},
):
    img_converted = np.frombuffer(img.getbuffer(), np.uint)

    # Populate required values for file meta information
    meta = pydicom.FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("", {}, preamble=b"\0" * 128)
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID

    ds.PatientName = patient_data["PatientName"]
    ds.PatientID = patient_data["PatientID"]
    ds.ImageComments = patient_data["ImageComments"]

    ds.Modality = "CT"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()

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

    validate_file_meta(ds.file_meta, enforce_standard=True)

    ds.PixelData = img_converted.tobytes()

    ds.save_as(file_name, write_like_original=False)


def read_dicom_file(file_name, tab):
    dcm_data = pydicom.dcmread(f"input/{file_name}")
    print(dcm_data)
    # Dane ogóle o pacjencie
    if "PatientID" not in dcm_data.dir():
        dcm_data.PatientID = ""
    dcm_data.PatientID = tab.text_input("PatientID", dcm_data.PatientID)
    if "PatientName" not in dcm_data.dir():
        dcm_data.PatientName = ""
    dcm_data.PatientName = tab.text_input("PatientName", dcm_data.PatientName)
    if "PatientAge" not in dcm_data.dir():
        dcm_data.PatientAge = ""
    dcm_data.PatientAge = tab.text_input("PatientAge", dcm_data.PatientAge)
    if "PatientSex" not in dcm_data.dir():
        dcm_data.PatientSex = ""
    dcm_data.PatientSex = tab.text_input("PatientSex", dcm_data.PatientSex)
    if "PatientBirthDate" not in dcm_data.dir():
        dcm_data.PatientBirthDate = ""
    dcm_data.PatientBirthDate = tab.text_input(
        "PatientBirthDate", dcm_data.PatientBirthDate
    )
    # Data badania
    if "StudyDate" not in dcm_data.dir():
        dcm_data.StudyDate = ""
    dcm_data.StudyDate = tab.text_input("StudyDate", dcm_data.StudyDate)
    # Komentarze
    if "ImageComments" not in dcm_data.dir():
        dcm_data.ImageComments = ""
    dcm_data.ImageComments = tab.text_input("ImageComments", dcm_data.ImageComments)

    print(dcm_data.PatientName)

    # img = dcm_data.pixel_array
    # print(img)
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
