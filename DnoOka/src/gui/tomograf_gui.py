from lib import tomograf
import numpy as np
from PIL import Image, ImageOps


def view_sliders(st):
    krok_ukladu = st.slider("Krok układu emiter/detektor:", 0.5, 10.0, 0.5)
    liczba_detektorów = st.slider(
        "Liczba detektorów dla jednego układu emiter/detektor", 90, 1000, 180, 90
    )
    rozwartosc = st.slider("Rozwartość/rozpiętość układu emiter/detektor:", 0, 360, 90)
    wyswietl_etapy_posrednie = st.checkbox("Wyświetl etapy pośrednie")
    filtrowanie = st.checkbox("Filtrowanie przez konwolucję")
    auto_contrast = st.checkbox("Auto kontrast")

    krok_skanowania = 0
    krok_odtwarzania = 0
    if wyswietl_etapy_posrednie:
        krok_skanowania = st.slider(
            "Krok skanowania:", 0.0, 360.0, krok_ukladu, krok_ukladu
        )
        if krok_skanowania == 360.0:
            krok_odtwarzania = st.slider(
                "Krok odtwarzania:", 0.0, 360.0, krok_ukladu, krok_ukladu
            )

    return (
        krok_ukladu,
        liczba_detektorów,
        rozwartosc,
        wyswietl_etapy_posrednie,
        filtrowanie,
        auto_contrast,
        krok_skanowania,
        krok_odtwarzania,
    )


def view_tomograf(
    st,
    image,
    krok_ukladu,
    liczba_detektorów,
    rozwartosc,
    wyswietl_etapy_posrednie,
    filtrowanie,
    auto_contrast,
    krok_skanowania,
    krok_odtwarzania,
):
    st.image(image, "Obraz źródłowy", clamp=True)

    if not wyswietl_etapy_posrednie:
        krok_skanowania = 360

    sinogram = tomograf.radon(
        image, krok_ukladu, liczba_detektorów, rozwartosc, krok_skanowania, filtrowanie
    )

    if filtrowanie:
        col1, col2, col3 = st.columns(3)
        # col1.image(sinogram, "Sinogram clamped by st", clamp=True)
        s1 = sinogram.copy()
        s1[sinogram < 0] = 0
        col1.image(s1, "Sinogram z odcięciem wartości < 0")

        m = sinogram.min()
        slope = 1 / (1 - m)

        def f(x):
            return slope * (x - m)

        s2 = np.vectorize(f)(sinogram.copy())
        col2.image(s2, "Sinogram po przemapowaniu wartości (odcienie szarości)")

        s3 = np.zeros((sinogram.shape[0], sinogram.shape[1], 3))
        mask = sinogram >= 0
        s3[mask, :] = sinogram[mask, np.newaxis]
        s3[~mask, 0] = np.vectorize(f)(sinogram[~mask])
        col3.image(s3, "Sinogram po przemapowaniu wartości (kolory)")
    else:
        st.image(sinogram, "Sinogram")

    if not wyswietl_etapy_posrednie:
        krok_odtwarzania = 360

    reconstructed = np.clip(
        tomograf.inverse_radon(
            sinogram,
            image.shape,
            krok_ukladu,
            liczba_detektorów,
            rozwartosc,
            krok_odtwarzania,
        ),
        0,
        1,
    )

    if auto_contrast:
        reconstructed_conv = (255 * reconstructed).astype(np.uint8)
        reconstructed_adj = ImageOps.autocontrast(
            Image.fromarray(
                reconstructed_conv,
                mode="L",
            ),
            0.1,
        )
        reconstructed = np.asarray(reconstructed_adj) / 255

    st.image(reconstructed, "Obraz po odtworzeniu")

    return reconstructed
