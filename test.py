import os
import numpy as np
import cv2
import dash
from dash import dcc, html, Output, Input
import plotly.express as px
import base64
import io
import matplotlib.pyplot as plt
from skimage.transform import resize

# Set image directory
IMAGE_DIR = "img"
images = [img for img in os.listdir(IMAGE_DIR) if img.endswith(('png', 'jpg', 'jpeg'))]

# Bresenham's line algorithm
def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

# Function to compute Radon transform
def radon_transform(image, angles, num_detectors, span):
    h, w = image.shape
    sinogram = np.zeros(num_detectors, (len(angles)))
    center = (w // 2, h // 2)
    
    for i, angle in enumerate(angles):
        angle_rad = np.deg2rad(angle)
        
        for d in range(num_detectors):
            offset = (d - num_detectors / 2) * (span / num_detectors)
            x0, y0 = int(center[0] + offset * np.cos(angle_rad)), int(center[1] + offset * np.sin(angle_rad))
            x1, y1 = int(center[0] - offset * np.cos(angle_rad)), int(center[1] - offset * np.sin(angle_rad))
            
            points = bresenham_line(x0, y0, x1, y1)
            sinogram[i, d] = sum(image[y, x] for x, y in points if 0 <= x < w and 0 <= y < h)
    
    return sinogram

# Function to perform inverse Radon transform (backprojection)
def inverse_radon_transform(sinogram, angles, image_shape):
    h, w = image_shape
    reconstructed = np.zeros((h, w))
    center = (w // 2, h // 2)
    
    for i, angle in enumerate(angles):
        angle_rad = np.deg2rad(angle)
        
        for d in range(sinogram.shape[1]):
            offset = (d - sinogram.shape[1] / 2) * (w / sinogram.shape[1])
            x0, y0 = int(center[0] + offset * np.cos(angle_rad)), int(center[1] + offset * np.sin(angle_rad))
            x1, y1 = int(center[0] - offset * np.cos(angle_rad)), int(center[1] - offset * np.sin(angle_rad))
            
            points = bresenham_line(x0, y0, x1, y1)
            value = sinogram[i, d] / len(points)
            for x, y in points:
                if 0 <= x < w and 0 <= y < h:
                    reconstructed[y, x] += value
    
    return reconstructed

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(id='image-selector', options=[{'label': img, 'value': img} for img in images], value=images[0]),
    dcc.Slider(id='angle-step', min=1, max=10, step=1, value=5, marks={i: str(i) for i in range(1, 11)}),
    dcc.Slider(id='num-detectors', min=10, max=200, step=10, value=50, marks={i: str(i) for i in range(10, 201, 50)}),
    dcc.Slider(id='span', min=50, max=300, step=10, value=200, marks={i: str(i) for i in range(50, 301, 50)}),
    dcc.Graph(id='sinogram-display'),
    dcc.Graph(id='reconstruction-display')
])

@app.callback(
    [Output('sinogram-display', 'figure'), Output('reconstruction-display', 'figure')],
    [Input('image-selector', 'value'), Input('angle-step', 'value'), Input('num-detectors', 'value'), Input('span', 'value')]
)
def update_images(image_name, angle_step, num_detectors, span):
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    
    angles = np.arange(0, 180, angle_step)
    sinogram = radon_transform(image, angles, num_detectors, span)
    reconstructed = inverse_radon_transform(sinogram, angles, (h, w))
    
    sinogram_fig = px.imshow(sinogram, aspect='auto', color_continuous_scale='gray')
    reconstruction_fig = px.imshow(reconstructed, color_continuous_scale='gray')
    
    return sinogram_fig, reconstruction_fig

if __name__ == '__main__':
    app.run(debug=True)
