import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tomograf
import pydicom
from PIL import Image
import cv2


# Streamlit app
st.title("Tomograf")

st.write("flkjaslkdj")

# Slider to adjust the parameter for image generation
slider_value = st.slider("Adjust the slider:", 0, 100, 50)

# Generate the image based on slider value
# image = generate_image(slider_value)


# Display the generated image using matplotlib
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# contents = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
# contents = Image.open()
# pimg = Image.open("img/Kropka.jpg")
# pimg =
# image = cv2.cvt(, cv2.IMREAD_GRAYSCALE)
image = cv2.imread("img/Kropka.jpg", cv2.IMREAD_GRAYSCALE)
ax1.imshow(image, cmap="gray")
ax1.axis("off")
sinogram = tomograf.radon(image, 2, 1000)
# sinogram = tomograf.radom_full(image, image.shape)
ax2.imshow(sinogram, cmap="gray")
ax2.axis("off")
reconstructed = tomograf.inverse_radon(sinogram, image.shape, 2, 1000)
# reconstructed = tomograf.radom_full(image, image.shape, inverse=True)
ax3.imshow(reconstructed, cmap="gray")
ax3.axis("off")
st.pyplot(fig)
