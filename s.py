import streamlit as st
import matplotlib.pyplot as plt
import tomograf
import pydicom
from PIL import Image
import cv2


def display_img(img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    st.pyplot(fig)


st.title("Tomograf")

krok_ukladu = st.slider("Krok układu emiter/detektor:", 1, 10, 1)
liczba_detektorów = st.slider(
    "Liczba detektorów dla jednego układu emiter/detektor", 1, 500, 180
)
rozwartosc = st.slider("Rozwartość/rozpiętość układu emiter/detektor:", 0, 180, 90)
wyswietl_etapy_posrednie = st.checkbox("Wyświetl etapy pośrednie")
filtrowanie = st.checkbox("Filtrowanie przez konwolucję")

krok_skanowania = 0
krok_odtwarzania = 0
if wyswietl_etapy_posrednie:
    krok_skanowania = st.slider("Krok skanowania:", 0, 360, krok_ukladu, krok_ukladu)
    if krok_skanowania == 360:
        krok_odtwarzania = st.slider(
            "Krok odtwarzania:", 0, 360, krok_ukladu, krok_ukladu
        )

image = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
display_img(image)

if wyswietl_etapy_posrednie:
    sinogram = tomograf.radon(
        image, krok_ukladu, liczba_detektorów, rozwartosc, krok_skanowania
    )
    display_img(sinogram)
else:
    sinogram = tomograf.radon(image, krok_ukladu, liczba_detektorów, rozwartosc)
    display_img(sinogram)


if wyswietl_etapy_posrednie:
    reconstructed = tomograf.inverse_radon(
        sinogram,
        image.shape,
        krok_ukladu,
        liczba_detektorów,
        rozwartosc,
        krok_odtwarzania,
    )
    display_img(reconstructed)
else:
    reconstructed = tomograf.inverse_radon(
        sinogram,
        image.shape,
        krok_ukladu,
        liczba_detektorów,
        rozwartosc,
    )
    display_img(reconstructed)
