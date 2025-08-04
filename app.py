# app.py
import streamlit as st
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
@st.cache_resource
def load_model_and_cascade():
    model = load_model("mask_detector.keras")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, face_cascade

model, face_cascade = load_model_and_cascade()

# Distance Estimation Constants
KNOWN_DISTANCE = 50  # cm
KNOWN_WIDTH = 14     # cm
FOCAL_LENGTH = 580

def estimate_distance(focal_length, known_width, width_in_frame):
    return (known_width * focal_length) / width_in_frame

# WebRTC Video Processor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = face_cascade.detectMultiScale(img, 1.1, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            resized = cv2.resize(face, (100, 100))
            input_img = preprocess_input(resized)
            input_img = np.expand_dims(input_img, axis=0)
            prediction = model.predict(input_img, verbose=0)
            label = np.argmax(prediction)
            label_text = "Mask" if label == 0 else "No Mask"
            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            distance = estimate_distance(FOCAL_LENGTH, KNOWN_WIDTH, w)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, f"{int(distance)} cm", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI
st.set_page_config(page_title="Mask Detection Web", layout="centered")

st.markdown("<h1 style='text-align:center;'>😷 Mask Detection System</h1>", unsafe_allow_html=True)

webrtc_streamer(key="mask-detect", video_processor_factory=VideoProcessor)
