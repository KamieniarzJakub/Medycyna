from pydicom.dataset import Dataset


def dicom_file_gui(tab, dcm_data: Dataset):
    DicomGUI = {
        "PatientID": tab.text_input("PatientID", dcm_data.get("PatientID", "")),
        "PatientName": tab.text_input("PatientName", dcm_data.get("PatientName", "")),
        "PatientAge": tab.text_input("PatientAge", dcm_data.get("PatientAge", "")),
        "PatientSex": tab.text_input("PatientSex", dcm_data.get("PatientSex", "")),
        "PatientBirthDate": tab.text_input(
            "PatientBirthDate", dcm_data.get("PatientBirthDate", "")
        ),
        "StudyDate": tab.text_input("StudyDate", dcm_data.get("StudyDate", "")),
        "ImageComments": tab.text_input(
            "ImageComments", dcm_data.get("ImageComments", "")
        ),
    }
    return DicomGUI


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
