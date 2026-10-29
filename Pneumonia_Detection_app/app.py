import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Classification",
    page_icon=None,
    layout="centered"
)

# --------------------------------------------------
# Custom styling: simple black background
# --------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }

        h1, h2, h3, p, label, span {
            color: #FFFFFF;
        }

        .stButton > button {
            background-color: #FFFFFF;
            color: #000000;
            border: none;
            border-radius: 5px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            width: 100%;
        }

        .stButton > button:hover {
            background-color: #D9D9D9;
            color: #000000;
        }

        [data-testid="stFileUploader"] {
            background-color: #111111;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 15px;
        }

        [data-testid="stFileUploader"] section {
            background-color: #111111;
        }

        [data-testid="stAlert"] {
            background-color: #111111;
        }

        hr {
            border-color: #333333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL_PATH = "pneumonia_mobilenetv2_finetuned.keras"
IMAGE_SIZE = (224, 224)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        safe_mode=False
    )

try:
    model = load_model()
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()

# --------------------------------------------------
# Application heading
# --------------------------------------------------

st.title("Pneumonia Classification")
st.write(
    "Chest X-ray image classification using a fine-tuned "
    "MobileNetV2 deep-learning model."
)

st.markdown("---")

# --------------------------------------------------
# Image upload
# --------------------------------------------------

st.subheader("Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Select an image in JPG, JPEG, or PNG format",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded chest X-ray",
            use_container_width=True
        )

        st.markdown("")

        if st.button("Classify Image"):

            with st.spinner("Analysing the image..."):

                resized_image = image.resize(IMAGE_SIZE)

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                prediction = model.predict(
                    image_array,
                    verbose=0
                )
                st.write("Raw model output:", prediction.tolist())
                st.write("Input shape:", image_array.shape)
                st.write("Minimum pixel value:", float(image_array.min()))
                st.write("Maximum pixel value:", float(image_array.max()))

                pneumonia_probability = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

                # Assumes class label 1 represents PNEUMONIA
                if pneumonia_probability >= 0.5:
                    predicted_class = "PNEUMONIA"
                    confidence = pneumonia_probability
                else:
                    predicted_class = "NORMAL"
                    confidence = 1.0 - pneumonia_probability

            st.markdown("---")
            st.subheader("Prediction Result")

            st.write(f"**Predicted class:** {predicted_class}")
            st.write(f"**Model score:** {confidence:.2%}")

            st.caption(
                "The model score is not a medically validated "
                "probability of disease."
            )

    except Exception:
        st.error(
            "The uploaded file could not be processed. "
            "Please upload a valid chest X-ray image."
        )

# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.markdown("---")

st.subheader("Disclaimer")

st.write(
    "This application is developed for educational and "
    "experimental purposes. Its predictions are not a medical "
    "diagnosis and must not be used to make healthcare decisions. "
    "Please consult a qualified healthcare professional for "
    "medical evaluation."
)
