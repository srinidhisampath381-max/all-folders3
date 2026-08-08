import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import os
import pyttsx3  # For text-to-speech
from gtts import gTTS
from io import BytesIO

# Initialize text-to-speech engine
engine = pyttsx3.init()

def text_to_speech(text):
    if text == "":
        text = "hello"
    tts = gTTS(text)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

st.title("🖐 Sign Language Detection with Face Mesh")

# Sidebar options
st.sidebar.header("Parameters")
enable_detection = st.sidebar.checkbox("Enable Detection")
start_button = st.sidebar.button("Start")

# Gesture labels
gesture_labels = {0: "hello", 1: "thank you", 2: "i love you"}

# Check if gesture model exists
model_path = "gesture_model.h5"
if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found. Please check the file path.")

# Initialize MediaPipe Hands & Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh()

# Set up webcam
camera_index = 0  # Change to 1 if webcam fails
cap = cv2.VideoCapture(camera_index)

if start_button:
    if not cap.isOpened():
        st.error("Failed to access webcam. Please check permissions and try again.")
    else:
        st.success("Webcam accessed successfully. Processing...")

        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        audio_placeholder = st.empty()

        while cap.isOpened() and enable_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_copy = frame.copy()  # Fix for cv2.line error

            results_hands = hands.process(frame)
            results_face = face_mesh.process(frame)

            # Draw face mesh landmarks
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        frame_copy, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )

            # Draw hand landmarks
            detected_gesture = "No gesture detected"
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Example logic for detecting "Thumbs Up" gesture
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    if thumb_tip.y < index_tip.y:  # If thumb is above index finger
                        detected_gesture = gesture_labels.get(0, "Unknown Gesture")

            # Display frame & results
            frame_placeholder.image(frame_copy, channels="RGB")
            result_placeholder.subheader(f"Detected Gesture: {detected_gesture}")

            # Speak the detected gesture
            if detected_gesture != "No gesture detected":
                audio_fp = text_to_speech(detected_gesture)
                audio_placeholder.audio(audio_fp, format='audio/mp3')

        cap.release()