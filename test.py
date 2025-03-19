import os
import numpy as np
import cv2
import dash
from dash import dcc, html, Output, Input, Dash, State, callback
import plotly.express as px
import base64
import io
import matplotlib.pyplot as plt
from skimage.transform import resize
import datetime

app = Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)

app.layout = html.Div(
    [
        dcc.Upload(
            id="upload-image",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        html.Div(id="output-image-upload"),
    ]
)


def parse_contents(contents, filename, date):
    return html.Div(
        [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html.Img(src=contents),
            html.Hr(),
            html.Div("Raw Content"),
            html.Pre(
                contents[0:200] + "...",
                style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
            ),
        ]
    )


#
@callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    State("upload-image", "last_modified"),
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d)
            for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children


#
# # Set image directory
# IMAGE_DIR = "img"
# images = [img for img in os.listdir(IMAGE_DIR) if img.endswith(("png", "jpg", "jpeg"))]
#
# app.layout = html.Div(
#     [
#         dcc.Dropdown(
#             id="image-selector",
#             options=[{"label": img, "value": img} for img in images],
#             value=images[0],
#         ),
#         dcc.Slider(
#             id="angle-step",
#             min=1,
#             max=10,
#             step=1,
#             value=5,
#             marks={i: str(i) for i in range(1, 11)},
#         ),
#         dcc.Slider(
#             id="num-detectors",
#             min=10,
#             max=200,
#             step=10,
#             value=50,
#             marks={i: str(i) for i in range(10, 201, 50)},
#         ),
#         dcc.Slider(
#             id="span",
#             min=50,
#             max=300,
#             step=10,
#             value=200,
#             marks={i: str(i) for i in range(50, 301, 50)},
#         ),
#         dcc.Graph(id="sinogram-display"),
#         dcc.Graph(id="reconstruction-display"),
#     ]
# )
#
#
# @app.callback(
#     [Output("sinogram-display", "figure"), Output("reconstruction-display", "figure")],
#     [
#         Input("image-selector", "value"),
#         Input("angle-step", "value"),
#         Input("num-detectors", "value"),
#         Input("span", "value"),
#     ],
# )
# def update_images(image_name, angle_step, num_detectors, span):
#     image_path = os.path.join(IMAGE_DIR, image_name)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     h, w = image.shape
#
#     angles = np.arange(0, 180, angle_step)
#     sinogram = radon_transform(image, angles, num_detectors, span)
#     reconstructed = inverse_radon_transform(sinogram, angles, (h, w))
#
#     sinogram_fig = px.imshow(sinogram, aspect="auto", color_continuous_scale="gray")
#     reconstruction_fig = px.imshow(reconstructed, color_continuous_scale="gray")
#
#     return sinogram_fig, reconstruction_fig
#

if __name__ == "__main__":
    app.run(debug=True)
