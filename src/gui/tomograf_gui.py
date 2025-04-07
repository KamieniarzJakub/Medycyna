from lib import tomograf
import numpy.typing as npt


def view_tomograf(st, image):
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

    st.image(image, "Obraz źródłowy")

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
    st.image(sinogram, "Sinogram")

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
        reconstructed[reconstructed < 0] = 0
    else:
        reconstructed = tomograf.inverse_radon(
            sinogram,
            image.shape,
            krok_ukladu,
            liczba_detektorów,
            rozwartosc,
        )
        reconstructed[reconstructed < 0] = 0

    st.image(reconstructed, "Obraz po odtworzeniu")
