import streamlit as st
import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Page Styling ---
st.set_page_config(page_title="Mask Detection", layout="centered")

# Title
st.markdown(
    """
    <div style='margin-top:-60px;margin-bottom:30px; display: flex; align-items: center; justify-content: center; height: 100px;'>
        <h1 style='margin: 0;'>Mask Detection System 😷</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom Button Styling
st.markdown("""
    <style>
    div.stButton > button {
        border: 2px solid #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 1.1em;
        font-weight: 600;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        border-color:red;
        color:white;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants for Distance Estimation ---
KNOWN_DISTANCE = 50  # in cm
KNOWN_WIDTH = 14     # in cm (approximate face width)
FOCAL_LENGTH = 580   # calibrated for your camera

def estimate_distance(focal_length, known_width, width_in_frame):
    return (known_width * focal_length) / width_in_frame

# --- Load Model and Face Detector ---
@st.cache_resource
def load_detection_resources():
    if not os.path.exists("mask_detector.keras"):
        st.error("Model file 'mask_detector.keras' not found.")
        st.stop()

    try:
        model = load_model("mask_detector.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, face_cascade

model, face_cascade = load_detection_resources()

# --- UI Controls ---
if 'run' not in st.session_state:
    st.session_state.run = False

col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
with col2:
    if st.button("▶ Start Detection"):
        st.session_state.run = True
with col4:
    if st.button("⏹ Stop Detection"):
        st.session_state.run = False

# --- Webcam Stream ---
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Webcam not accessible.")
        break

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        distance = estimate_distance(FOCAL_LENGTH, KNOWN_WIDTH, w)
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)
        label = np.argmax(prediction)

        label_text = "Mask" if label == 0 else "No Mask"
        color = (0, 255, 0) if label == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"{int(distance)} cm", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

    time.sleep(0.03)

cap.release()
frame_placeholder.empty()
