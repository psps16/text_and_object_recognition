import streamlit as st
import cv2
import numpy as np
import pytesseract
from gtts import gTTS
import os
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up Tesseract command path (update this path)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust for Streamlit Cloud

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.previous_detected_objects = []
        self.text_output = ""
        self.detected_objects = []

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")

        # Detect objects using YOLO
        results = model(frame)
        annotated_frame = frame.copy()
        detected_objects = []

        if results:
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

        # Recognize text using Tesseract
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_frame)

        # Update the class attributes to display in Streamlit
        self.detected_objects = detected_objects
        self.text_output = text

        return annotated_frame

    def speak_text(self, text):
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.save(f.name)
            return f.name

# Streamlit page layout
st.title("Text and object detection Assistant")
st.write("This app performs real-time object detection and text recognition.")

# Instantiate the video processor and start the WebRTC streamer
ctx = webrtc_streamer(key="example", video_transformer_factory=VideoProcessor)

if ctx.video_transformer:
    # Display detected objects and text below the video stream
    st.write("Detected Objects:")
    if ctx.video_transformer.detected_objects:
        st.write(", ".join(ctx.video_transformer.detected_objects))
        audio_file = ctx.video_transformer.speak_text(f"Detected objects: {', '.join(ctx.video_transformer.detected_objects)}")
        st.audio(audio_file, format='audio/mp3')
        os.remove(audio_file)

    st.write("Detected Text:")
    if ctx.video_transformer.text_output:
        st.write(ctx.video_transformer.text_output)
        audio_file = ctx.video_transformer.speak_text(ctx.video_transformer.text_output)
        st.audio(audio_file, format='audio/mp3')
        os.remove(audio_file)
