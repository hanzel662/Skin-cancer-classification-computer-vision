import joblib
import numpy as np
from PIL import Image
import streamlit as st
import os

from preprocess_file import (
    dullrazor,
    denoise,
    normalize_color,
    segment_otsu,
    extract_features
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "output",
    "skin_cancer_ensemble.joblib"
)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
le = bundle["label_encoder"]

st.title("Skin Cancer Detection")
st.caption("A Computer Vision–based lesion classification project.")
st.warning("Educational use only. Not a medical diagnosis.")

uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if "history" not in st.session_state:
    st.session_state.history = []

if "mask" not in st.session_state:
    st.session_state.mask = None

if uploaded is not None:
    col1, col2 = st.columns(2)

    # -------------------------------
    # Display uploaded image
    # -------------------------------
    with col1:
        image = Image.open(uploaded).convert("RGB")
        st.subheader("Uploaded Image")
        st.image(image)

    # -------------------------------
    # Process & predict
    # -------------------------------
    with col2:
        st.subheader("Analysis")

        if st.button("Run Detection"):
            with st.spinner("Processing image..."):

                img = np.array(image)

                # Preprocessing
                img = dullrazor(img)
                img = denoise(img)
                img = normalize_color(img)
                mask = segment_otsu(img)

                # Feature extraction
                features = extract_features(img, mask)
                features = scaler.transform([features])

                # Prediction
                pred = model.predict(features)[0]
                label = le.inverse_transform([pred])[0]

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    confidence = np.max(probs) * 100
                else:
                    probs = None
                    confidence = None

                # Save mask + result to session state
                st.session_state.mask = mask

                st.session_state.history.append({
                    "image": image.copy(),
                    "label": label,
                    "confidence": confidence
                })

            label_map = {
                "bcc": "Basal Cell Carcinoma",
                "mel": "Melanoma",
                "nv": "Melanocytic Nevus",
                "bkl": "Benign Keratosis-like Lesion",
                "akiec": "Actinic Keratosis / Bowen's Disease",
                "df": "Dermatofibroma",
                "vasc": "Vascular Lesion"
            }

            label_name = label_map.get(label, label)

            st.success(f"Prediction: **{label_name.upper()}**")

            if confidence:
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.2f}%"
                )

            if label in ["mel"]:
                st.error("High-risk lesion.")
            elif label in ["bcc", "akiec"]:
                st.warning("Medium-risk lesion.")
            else:
                st.success("Low-risk lesion.")

st.divider()
st.subheader("Prediction History")

if len(st.session_state.history) == 0:
    st.info("No predictions yet.")
else:
    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.container():
            cols = st.columns([1, 2])

            with cols[0]:
                st.image(item["image"], width="stretch")

            with cols[1]:
                label = item["label"]

                label_map = {
                    "bcc": "Basal Cell Carcinoma",
                    "mel": "Melanoma",
                    "nv": "Melanocytic Nevus",
                    "bkl": "Benign Keratosis-like Lesion",
                    "akiec": "Actinic Keratosis / Bowen's Disease",
                    "df": "Dermatofibroma",
                    "vasc": "Vascular Lesion"
                }

                label_name = label_map.get(label, label)

                st.write(f"Label: **{label_name.upper()}**")

                if item["confidence"] is not None:
                    st.write(f"Confidence: **{item['confidence']:.2f}%**")

                if label in ["mel"]:
                    st.error("High-risk lesion.")
                elif label in ["bcc", "akiec"]:
                    st.warning("Medium-risk lesion.")
                else:
                    st.success("Low-risk lesion.")

# Optional: clear history
if st.button("Clear History"):
    st.session_state.history.clear()