import streamlit as st
from ultralytics import YOLO
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n-face.pt")  # Make sure this file is in the same directory

model = load_model()

# Streamlit UI
st.title("🧠 Real-Time Face Detection with YOLOv8")
st.markdown("This app uses YOLOv8 to detect faces from your webcam feed in real-time.")

# Webcam detection class
class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(source=img, save=False, verbose=False)
        annotated = results[0].plot()
        return annotated

# Start streamlit-webrtc webcam
webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
