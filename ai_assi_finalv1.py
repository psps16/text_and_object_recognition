import streamlit as st
import cv2
import numpy as np
import pytesseract
from gtts import gTTS
import os
import tempfile
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set up Tesseract command path (update to your path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

def detect_objects(frame):
    try:
        results = model(frame)
        return results
    except Exception as e:
        st.error(f"Error in object detection: {str(e)}")
        return None

def recognize_text(frame):
    text = pytesseract.image_to_string(frame)
    return text

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        tts.save(f.name)
        return f.name

st.title("AI Assistant for Blind people")
st.write("This app performs real-time object detection and text recognition.")

# Initialize state
if 'previous_detected_objects' not in st.session_state:
    st.session_state.previous_detected_objects = []
if 'run' not in st.session_state:
    st.session_state.run = False

# Real-time video processing (camera input)
st.write("Real-time Video Processing")

start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Unable to access the camera")
else:
    if start_button:
        st.session_state.run = True

    if stop_button:
        st.session_state.run = False
        cap.release()
        cv2.destroyAllWindows()

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        # Detect objects
        results = detect_objects(frame)
        if results:
            annotated_frame = frame.copy()
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

        # Recognize text
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = recognize_text(gray_frame)

        # Display the frame
        FRAME_WINDOW.image(annotated_frame, channels="BGR")

        # Provide audio feedback for detected objects and text
        if text:
            st.write(f"Detected Text: {text}")
            audio_file = speak_text(text)
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')
            os.remove(audio_file)

        if detected_objects and detected_objects != st.session_state.previous_detected_objects:
            object_text = ", ".join(detected_objects)
            st.write(f"Detected Objects: {object_text}")
            audio_file = speak_text(f"Detected objects: {object_text}")
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')
            os.remove(audio_file)
            st.session_state.previous_detected_objects = detected_objects

        if not st.session_state.run:
            break

    cap.release()
    cv2.destroyAllWindows()

st.write("Stop the camera by clicking the 'Stop Detection' button.")
st.markdown("---")
st.write("Developed by [Pranav bhat and Siddarth B S]")
