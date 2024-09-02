import streamlit as st
import cv2
import numpy as np
import pytesseract
from gtts import gTTS
import os
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up Tesseract command path (update to your path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Object detection
        results = self.model(img)
        annotated_frame = img.copy()
        detected_objects = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                cls = self.model.names[int(box.cls)]
                detected_objects.append(cls)
                annotated_frame = cv2.rectangle(
                    annotated_frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    (0, 255, 0), 2
                )
                annotated_frame = cv2.putText(
                    annotated_frame, 
                    cls, 
                    (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (36, 255, 12), 
                    2
                )

        # Text recognition
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_frame)

        return annotated_frame

def recognize_text(frame):
    text = pytesseract.image_to_string(frame)
    return text

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        tts.save(f.name)
        return f.name

st.title("AI Audio Assistant")
st.write("This app performs real-time object detection and text recognition.")

ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Handle detected objects and text output
if ctx.video_processor:
    processor = ctx.video_processor
    if processor:
        detected_objects = processor.detected_objects
        if detected_objects:
            object_text = ", ".join(detected_objects)
            st.write(f"Detected Objects: {object_text}")
            audio_file = speak_text(f"Detected objects: {object_text}")
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')
            os.remove(audio_file)
