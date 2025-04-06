import streamlit as st
import matplotlib.pyplot as plt
import tomograf
import pydicom
from PIL import Image
import cv2


st.title("Tomograf")

krok_ukladu = st.slider("Krok układu emiter/detektor:", 1, 10, 1)
liczba_detektorów = st.slider(
    "Liczba detektorów dla jednego układu emiter/detektor", 1, 500, 180
)
rozwartosc = st.slider("Rozwartość/rozpiętość układu emiter/detektor:", 0, 180, 90)
wyswietl_etapy_posrednie = st.checkbox("Wyświetl etapy pośrednie")
filtrowanie = st.checkbox("Filtrowanie przez konwolucję")

if wyswietl_etapy_posrednie:
    krok_skanowania = st.slider("Krok skanowania:", 0.0, 360 / krok_ukladu)
    krok_odtwarzania = st.slider("Krok odtwarzania:", 0.0, 360 / krok_ukladu)

(fig1, ax1) = plt.subplots()
image = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
ax1.imshow(image, cmap="gray")
ax1.axis("off")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sinogram = tomograf.radon(image, krok_ukladu, liczba_detektorów, rozwartosc)
ax2.imshow(sinogram, cmap="gray")
ax2.axis("off")
st.pyplot(fig2)


fig3, ax3 = plt.subplots()
reconstructed = tomograf.inverse_radon(
    sinogram, image.shape, krok_ukladu, liczba_detektorów, rozwartosc
)
ax3.imshow(reconstructed, cmap="gray")
ax3.axis("off")
st.pyplot(fig3)
