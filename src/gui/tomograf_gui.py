from lib import tomograf
import numpy.typing as npt
import numpy as np


def view_sliders(st):
    krok_ukladu = st.slider("Krok układu emiter/detektor:", 1, 10, 1)
    liczba_detektorów = st.slider(
        "Liczba detektorów dla jednego układu emiter/detektor", 1, 500, 180
    )
    rozwartosc = st.slider("Rozwartość/rozpiętość układu emiter/detektor:", 0, 180, 90)
    wyswietl_etapy_posrednie = st.checkbox("Wyświetl etapy pośrednie")
    filtrowanie = st.checkbox("Filtrowanie przez konwolucję")

    krok_skanowania = 0
    krok_odtwarzania = 0
    if wyswietl_etapy_posrednie:
        krok_skanowania = st.slider(
            "Krok skanowania:", 0, 360, krok_ukladu, krok_ukladu
        )
        if krok_skanowania == 360:
            krok_odtwarzania = st.slider(
                "Krok odtwarzania:", 0, 360, krok_ukladu, krok_ukladu
            )

    return (
        krok_ukladu,
        liczba_detektorów,
        rozwartosc,
        wyswietl_etapy_posrednie,
        filtrowanie,
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
    krok_skanowania,
    krok_odtwarzania,
):
    st.image(image, "Obraz źródłowy", clamp=True)

    sinogram: npt.NDArray
    if wyswietl_etapy_posrednie:
        sinogram = tomograf.radon(
            image,
            krok_ukladu,
            liczba_detektorów,
            rozwartosc,
            krok_skanowania,
            filtrowanie,
        )
    else:
        sinogram = tomograf.radon(
            image, krok_ukladu, liczba_detektorów, rozwartosc, run_convolve=filtrowanie
        )

    col1, col2 = st.columns(2)
    col1.image(sinogram, "Sinogram clamped by st", clamp=True)

    m = sinogram.min()
    slope = 1 / (1 - m)

    def f(x):
        return slope * (x - m)

    s2 = np.vectorize(f)(sinogram.copy())
    col2.image(s2, "Sinogram remapped by numpy")

    reconstructed: npt.NDArray
    if wyswietl_etapy_posrednie:
        reconstructed = tomograf.inverse_radon(
            sinogram,
            image.shape,
            krok_ukladu,
            liczba_detektorów,
            rozwartosc,
            krok_odtwarzania,
        )
    else:
        reconstructed = tomograf.inverse_radon(
            sinogram,
            image.shape,
            krok_ukladu,
            liczba_detektorów,
            rozwartosc,
        )

    st.image(reconstructed, "Obraz po odtworzeniu", clamp=True)
