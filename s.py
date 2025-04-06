import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tomograf
import pydicom
from PIL import Image
import cv2


# Streamlit app
st.title("Tomograf")

# Slider to adjust the parameter for image generation
krok_ukladu = st.slider("Krok ∆α układu emiter/detektor:", 1, 10, 1)
liczba_detektorów = st.slider(
    "Dla jednego układu emiter/detektor liczbę detektorów (n):", 1, 500, 180
)
rozwartosc = st.slider("Rozwartość/rozpiętość układu emiter/detektor (l)", 0, 180, 90)

# Display the generated image using matplotlib
(fig1, ax1) = plt.subplots()
image = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
ax1.imshow(image, cmap="gray")
ax1.axis("off")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sinogram = tomograf.radon(image, krok_ukladu, liczba_detektorów, rozwartosc)
# sinogram = tomograf.radom_full(image, image.shape)
ax2.imshow(sinogram, cmap="gray")
ax2.axis("off")
st.pyplot(fig2)


fig3, ax3 = plt.subplots()
reconstructed = tomograf.inverse_radon(
    sinogram, image.shape, krok_ukladu, liczba_detektorów, rozwartosc
)
# reconstructed = tomograf.radom_full(image, image.shape, inverse=True)
ax3.imshow(reconstructed, cmap="gray")
ax3.axis("off")
st.pyplot(fig3)
