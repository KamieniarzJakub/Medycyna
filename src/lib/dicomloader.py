from typing import Dict
import pydicom
import numpy as np
from pydicom.dataset import Dataset, validate_file_meta
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.uid import CTImageStorage
from pydicom.uid import generate_uid


def create_DICOM(
    img: np.ndarray,
    patient_data: Dict[str, str] = {
        "PatientName": "",
        "PatientID": "",
        "ImageComments": "",
    },
):
    # Populate required values for file meta information
    meta = pydicom.FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset({}, preamble=b"\0" * 128)
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

    ds.Rows, ds.Columns = img.shape

    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    validate_file_meta(ds.file_meta, enforce_standard=True)

    ds.PixelData = img.tobytes()

    return ds
