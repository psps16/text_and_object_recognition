import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, ClientSettings
import cv2
import numpy as np
import pytesseract
from gtts import gTTS
import tempfile
import os
from ultralytics import YOLO
import av

st.set_page_config(page_title="AI Audio Assistant", page_icon="🤖")

st.title("🤖 AI Audio Assistant")
st.write("This app performs real-time object detection and text recognition.")

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')
    return model

model = load_model()

# Set up pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path if needed

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.previous_detected_objects = set()
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Object Detection
        results = self.model(img)
        annotated_frame = img.copy()
        detected_objects = set()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    confidence = box.conf[0]
                    label = self.model.names[cls_id]
                    detected_objects.add(label)

                    # Draw bounding box and label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Text Recognition
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        text = text.strip()

        # Audio Feedback for Detected Objects
        new_objects = detected_objects - self.previous_detected_objects
        if new_objects:
            object_text = ", ".join(new_objects)
            self.speak_text(f"Detected objects: {object_text}", "object_audio.mp3")
            audio_file = open("object_audio.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')
            audio_file.close()
            os.remove("object_audio.mp3")
            self.previous_detected_objects = detected_objects

        # Audio Feedback for Detected Text
        if text:
            self.speak_text(f"Detected text: {text}", "text_audio.mp3")
            audio_file = open("text_audio.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')
            audio_file.close()
            os.remove("text_audio.mp3")
            st.write(f"**Detected Text:** {text}")

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    @staticmethod
    def speak_text(text, filename):
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

# WebRTC Client Settings
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)

webrtc_ctx = webrtc_streamer(
    key="ai-audio-assistant",
    mode=WebRtcMode.SENDRECV,
    client_settings=WEBRTC_CLIENT_SETTINGS,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.write("Developed by [Pranav Bhat and Siddarth B S]")
