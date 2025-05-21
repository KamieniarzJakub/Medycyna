import numpy as np


# Dodawanie czarnej ramki do niekwadratowych zdjęć
def add_borders_to_rectangle(img):
    h, w = img.shape
    m = max(w, h)
    pw = (
        (
            np.floor((m - h) / 2, dtype=int, casting="unsafe"),
            np.ceil((m - h) / 2, dtype=int, casting="unsafe"),
        ),
        (
            np.floor((m - w) / 2, dtype=int, casting="unsafe"),
            np.ceil((m - w) / 2, dtype=int, casting="unsafe"),
        ),
    )
    return np.pad(
        img,
        pw,
        "constant",
        constant_values=(0),
    )
