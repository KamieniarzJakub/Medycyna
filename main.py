import os
import numpy as np
import cv2
from PIL import Image
import dash
from dash import dcc, html, Output, Input, Dash, State, callback
import plotly.express as px
import base64
import io
import matplotlib.pyplot as plt
from skimage.transform import resize
import datetime
import tomograf

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(id="output-image-upload"),
                        dcc.Upload(
                            id="upload-image",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select File")]
                            ),
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
                            multiple=False,
                        ),
                    ]
                ),
                dcc.Graph(id="sinogram-display"),
                dcc.Graph(id="reconstruction-display"),
            ],
            style={
                "display": "grid",
                "grid-template-columns": "repeat(2,1fr)",
                "gap": "10px",
            },
        ),
        html.Div(
            children=[
                "Paremeters",
                dcc.Checklist(["Show emitters"], [], id="show-emitters"),
                html.Div("Angle step"),
                dcc.Slider(
                    id="angle-step",
                    min=1,
                    max=10,
                    step=1,
                    value=5,
                ),
                html.Div("Number of detectors"),
                dcc.Slider(
                    id="num-detectors",
                    min=1,
                    max=50,
                    step=1,
                    value=20,
                ),
                html.Div("Detectors span (degrees)"),
                dcc.Slider(
                    id="span",
                    min=10,
                    max=100,
                    step=10,
                    value=200,
                ),
            ],
        ),
    ],
)


@callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
)
def parse_file(contents, filename):
    if contents is None:
        return None
    return html.Div(
        [
            html.H5(filename),
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


@app.callback(
    [
        Output("sinogram-display", "figure"),
        Output("reconstruction-display", "figure"),
    ],
    Input("upload-image", "contents"),
    # [
    #     Input("angle-step", "value"),
    #     Input("num-detectors", "value"),
    #     Input("span", "value"),
    # ],
)
def update_images(contents):
    if contents is not None:
        pimg = Image.open(contents)
        image = cv2.cvtColor(np.array(pimg), cv2.IMREAD_GRAYSCALE)
        h, w = image.shape

        sinogram = tomograf.radon(image)
        reconstructed = tomograf.radon(sinogram, (h, w), inverse=True)

        sinogram_fig = px.imshow(sinogram, aspect="auto", color_continuous_scale="gray")
        reconstruction_fig = px.imshow(reconstructed, color_continuous_scale="gray")

        return sinogram_fig, reconstruction_fig
    else:
        return (None, None)


if __name__ == "__main__":
    app.run(debug=True)
