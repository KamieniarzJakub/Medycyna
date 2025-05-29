import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from PIL import Image, UnidentifiedImageError
import gzip
import numpy as np
import io

from gui.dno_oka_gui import view_sliders, view_dno_oka
from lib.mse import calc_mse
from lib.przetwarzanie import preprocess_image, segment_vessels, postprocess_image, divide_image_into_chunks
from lib.calc_diff import visualize_array_difference
from lib.features import extract_features_from_patch
from lib.classifier import train_classifier, prepare_dataset, evaluate_classifier
from sklearn.metrics import classification_report


from sklearn.model_selection import train_test_split

supported_file_types = ["jpg", "jpeg", "png", "ppm", "gz", "webp", "avif", "gif"]

def read_img(fil: io.BytesIO | UploadedFile) -> Image.Image:
    try:
        return Image.open(fil).copy()
    except UnidentifiedImageError:
        if isinstance(fil, UploadedFile):
            raise Exception("Błędny typ pliku: ", fil.type, ", plik: ", fil.name, ", wspierane rozszerzenia: ", supported_file_types)
        else:
            raise Exception("Błędny typ pliku; plik: ", fil.name, ", wspierane rozszerzenia: ", supported_file_types)

def read_file(file: UploadedFile) -> Image.Image:
    if file.type in ["application/gzip", "application/x-gzip"]:
        with gzip.open(file, "rb") as f:
            return read_img(io.BytesIO(f.read()))
    else:
        return read_img(file)

st.title("Wykrywanie naczyń dna siatkówki oka")

file = st.file_uploader(
    "Skan siatkówki oka do analizy", type=supported_file_types, accept_multiple_files=False
)

expected_result = st.file_uploader(
    "[Opcjonalne] docelowy obraz naczyń (maska ekspercka)", type=supported_file_types, accept_multiple_files=False
)

if file is not None:
    image = read_file(file).convert("RGB")
    image_arr = np.asarray(image)

    st.image(image, caption="Oryginalny obraz", use_container_width=True, clamp=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Wstępne przetwarzanie", "Segmentacja naczyń", "Postprocessing", "Klasyfikacja ML (Random Forest)"])

    # Przetwarzanie obrazu
    pre = preprocess_image(image_arr / 255.0)
    vessels = segment_vessels(pre)
    final = postprocess_image(vessels)
    diff: np.ndarray | None = None

    expected_image: Image.Image | None = None
    if expected_result is not None:
        expected_image = read_file(expected_result)
        expected_image_arr = np.asarray(expected_image)
        diff = visualize_array_difference(final, expected_image_arr)

    with tab1:
        st.image(pre, caption="Po wstępnym przetwarzaniu", use_container_width=True, clamp=True)

    with tab2:
        st.image(vessels, caption="Segmentacja naczyń (Frangi)", use_container_width=True, clamp=True)

    with tab3:
        st.image(final, caption="Po końcowym przetwarzaniu", use_container_width=True, clamp=True)

        if diff is not None and expected_image is not None:
            st.image(expected_image, "Obraz docelowy", clamp=True)

            img = Image.fromarray(diff, 'RGB')
            st.image(img, "Różnica względem obrazu docelowego", clamp=True)
            """
            - Na czerwono nadmiarowe wykrycia
            - Na czarno brak różnicy, czyli poprawne
            - Na niebiesko brakujące wykrycia
            """

            mse_result = (diff ** 2).mean()
            st.text("Błąd średniokwadratowy: " + str(mse_result))

    with tab4:
        st.subheader("Klasyfikacja na podstawie cech wycinków (15x15)")

        if expected_result is None:
            st.warning("Aby skorzystać z klasyfikatora, należy załadować obraz docelowy (maskę ekspercką).")
        else:
            chunk_size = (15, 15)

            patches = divide_image_into_chunks(pre, chunk_size)
            labels_img_arr = np.asarray(expected_image.convert("L"))
            labels_chunks = divide_image_into_chunks(labels_img_arr > 200, chunk_size)


            features = []
            targets = []

            for patch_data, label_patch_data in zip(patches, labels_chunks):
                feat = extract_features_from_patch(patch_data)
                center_label = label_patch_data[label_patch_data.shape[0] // 2, label_patch_data.shape[1] // 2]
                features.append(feat)
                targets.append(int(center_label))

            features = np.array(features)
            targets = np.array(targets)

            if len(np.unique(targets)) < 2:
                st.warning("Maska zawiera tylko jedną klasę. Potrzebne są co najmniej dwie klasy (tło i naczynie).")
            else:
                X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, stratify=targets, random_state=42)
                X_bal, y_bal = prepare_dataset(X_train, y_train)

                st.write(f"Zbiór uczący (zbalansowany): {len(y_bal)} próbek ({np.sum(y_bal==1)} naczyń, {np.sum(y_bal==0)} tła)")
                st.write(f"Zbiór testowy: {len(y_test)} próbek ({np.sum(y_test==1)} naczyń, {np.sum(y_test==0)} tła)")

                clf = train_classifier(X_bal, y_bal)
                report = evaluate_classifier(clf, X_test, y_test)

                st.subheader("Raport klasyfikatora")
                st.json(report)

                preds = clf.predict(features)

                padded_height = int(np.ceil(pre.shape[0] / chunk_size[0]) * chunk_size[0])
                padded_width = int(np.ceil(pre.shape[1] / chunk_size[1]) * chunk_size[1])

                num_chunks_h = padded_height // chunk_size[0]
                num_chunks_w = padded_width // chunk_size[1]

                if num_chunks_w * num_chunks_h != len(preds):
                    st.error(f"Błąd: liczba przewidywanych chunków ({len(preds)}) nie zgadza się z ({num_chunks_h}x{num_chunks_w}={num_chunks_h * num_chunks_w})")
                    st.stop()

                # Rekonstrukcja maski z przewidywań
                pred_mask_chunks_reshaped = preds.reshape((num_chunks_h, num_chunks_w))
                full_mask = np.kron(pred_mask_chunks_reshaped, np.ones(chunk_size))
                full_mask = full_mask[:pre.shape[0], :pre.shape[1]]
                st.image(full_mask, caption="Maska wynikowa z klasyfikatora", use_container_width=True, clamp=True)

                # Obliczanie metryk dla maski ML w porównaniu do maski eksperckiej
                if expected_image is not None:
                    expected_binary_mask = (np.asarray(expected_image.convert("L")) > 200).astype(int)

                    if full_mask.shape != expected_binary_mask.shape:
                        st.warning("Rozmiary masek ML i eksperckiej różnią się. Dopasowuję...")
                        min_h = min(full_mask.shape[0], expected_binary_mask.shape[0])
                        min_w = min(full_mask.shape[1], expected_binary_mask.shape[1])
                        full_mask_cropped = full_mask[:min_h, :min_w]
                        expected_binary_mask_cropped = expected_binary_mask[:min_h, :min_w]
                    else:
                        full_mask_cropped = full_mask
                        expected_binary_mask_cropped = expected_binary_mask

                    ml_report = classification_report(expected_binary_mask_cropped.flatten(), full_mask_cropped.flatten(), output_dict=True)
                    st.subheader("Raport klasyfikatora ML (pełna maska vs ekspercka)")
                    st.json(ml_report)

                    ml_diff = visualize_array_difference(full_mask_cropped, expected_binary_mask_cropped)
                    ml_img = Image.fromarray(ml_diff, 'RGB')
                    st.image(ml_img, "Różnica względem obrazu docelowego (klasyfikator ML)", clamp=True)