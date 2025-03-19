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
        html.Div(
            [
                html.Div(
                    [
                        html.Img(id="output-image-upload"),
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
                html.Img(id="sinogram-display"),
                html.Img(id="reconstruction-display"),
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


# def parse_contents(contents, filename, date):
#     return html.Div(
#         [
#             html.H5(filename),
#             html.H6(datetime.datetime.fromtimestamp(date)),
#             # HTML images accept base64 encoded strings in the same format
#             # that is supplied by the upload
#             html.Img(src=contents),
#             html.Hr(),
#             html.Div("Raw Content"),
#             html.Pre(
#                 contents[0:200] + "...",
#                 style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
#             ),
#         ]
#     )


#
# @callback(
#     Output("output-image-upload", "children"),
#     Input("upload-image", "contents"),
#     State("upload-image", "filename"),
#     State("upload-image", "last_modified"),
# )
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d)
#             for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
#         ]
#         return children


# @app.callback(
#     [Output("sinogram-display", "src"), Output("reconstruction-display", "src")],
#     Input("upload-image", "contents"),
#     State("upload-image", "filename"),
#     State("upload-image", "last_modified"),
#     # [
#     #     Input("angle-step", "value"),
#     #     Input("num-detectors", "value"),
#     #     Input("span", "value"),
#     # ],
# )
# def update_images(contents, filename, last_modified):
#     if contents is not None:
#         # image_path = os.path.join(IMAGE_DIR, image_name)
#         image = cv2.imread(filename[0], cv2.IMREAD_GRAYSCALE)
#         # h, w = image.shape
#
#         # sinogram = radon_transform(image, angles, num_detectors, span)
#         # reconstructed = inverse_radon_transform(sinogram, angles, (h, w))
#
#         sinogram_fig = px.imshow(image, aspect="auto", color_continuous_scale="gray")
#         reconstruction_fig = px.imshow(image, color_continuous_scale="gray")
#
#         return sinogram_fig, reconstruction_fig
#

if __name__ == "__main__":
    app.run(debug=True)
