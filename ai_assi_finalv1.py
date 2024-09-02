import streamlit as st
import cv2
import numpy as np
import pytesseract
from gtts import gTTS
import os
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up Tesseract command path (update to your path)
pytesseract.pytesseract.tesseract_cmd = r'/app/.apt/usr/bin/tesseract'  # Adjust for Streamlit Cloud

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detected_objects = []
        self.text = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Object detection
        results = model(img)
        annotated_frame = img.copy()
        detected_objects = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                cls = model.names[int(box.cls)]
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
        self.detected_objects = detected_objects

        # Text recognition
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.text = pytesseract.image_to_string(gray_frame)
        
        return annotated_frame

    def get_text(self):
        return self.text

    def get_detected_objects(self):
        return self.detected_objects

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        tts.save(f.name)
        return f.name

st.title("AI Audio Assistant")
st.write("This app performs real-time object detection and text recognition.")

ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

if ctx.video_processor:
    processor = ctx.video_processor

    # Display detected objects and recognized text
    if processor:
        detected_objects = processor.get_detected_objects()
        if detected_objects:
            object_text = ", ".join(detected_objects)
            st.write(f"Detected Objects: {object_text}")
            audio_file = speak_text(f"Detected objects: {object_text}")
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')
            os.remove(audio_file)

        recognized_text = processor.get_text()
        if recognized_text:
            st.write(f"Detected Text: {recognized_text}")
            audio_file = speak_text(recognized_text)
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')
            os.remove(audio_file)
            
st.markdown("---")
st.write("Developed by [Pranav bhat and Siddarth B S]")
