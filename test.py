import os
import cv2
import numpy as np
import dash
from dash import dcc, html, Output, Input
import plotly.express as px
import base64
import io


# Set image directory
IMAGE_DIR = "img"
images = [img for img in os.listdir(IMAGE_DIR) if img.endswith(("png", "jpg", "jpeg"))]


# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Dropdown(
            id="image-selector",
            options=[{"label": img, "value": img} for img in images],
            value=images[0],
        ),
        dcc.Slider(
            id="brightness-slider",
            min=0.5,
            max=2,
            step=0.1,
            value=1,
            marks={i: str(i) for i in np.arange(0.5, 2.1, 0.5)},
        ),
        dcc.Slider(
            id="contrast-slider",
            min=0.5,
            max=2,
            step=0.1,
            value=1,
            marks={i: str(i) for i in np.arange(0.5, 2.1, 0.5)},
        ),
        dcc.Graph(id="image-display"),
    ]
)


def process_image(image_path, brightness=1.0, contrast=1.0):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=(brightness - 1) * 100)
    return img


@app.callback(
    Output("image-display", "figure"),
    [
        Input("image-selector", "value"),
        Input("brightness-slider", "value"),
        Input("contrast-slider", "value"),
    ],
)
def update_image(image_name, brightness, contrast):
    image_path = os.path.join(IMAGE_DIR, image_name)
    img = process_image(image_path, brightness, contrast)
    fig = px.imshow(img)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
