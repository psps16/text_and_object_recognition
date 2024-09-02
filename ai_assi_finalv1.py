import streamlit as st
import cv2
import numpy as np
import pytesseract
from gtts import gTTS
import os
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up Tesseract command path (update to your path)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust for Streamlit Cloud

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detected_objects = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.detect_objects(img)
        text = self.recognize_text(img)

        # Annotate the frame with detection results
        annotated_frame = img.copy()
        if results:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].numpy()
                    cls = model.names[int(box.cls)]
                    self.detected_objects.append(cls)
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

        return annotated_frame

    def detect_objects(self, frame):
        try:
            results = model(frame)
            return results
        except Exception as e:
            st.error(f"Error in object detection: {str(e)}")
            return None

    def recognize_text(self, frame):
        text = pytesseract.image_to_string(frame)
        return text

st.title("AI Audio Assistant")
st.write("This app performs real-time object detection and text recognition.")

ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

if ctx.video_processor:
    processor = ctx.video_processor
    detected_objects = processor.detected_objects
    if detected_objects:
        object_text = ", ".join(detected_objects)
        st.write(f"Detected Objects: {object_text}")
