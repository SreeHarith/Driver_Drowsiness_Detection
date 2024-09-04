import cv2
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import mediapipe as mp
import streamlit as st
import time

# Set Streamlit layout to wide and hide the sidebar
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Initialize Pygame mixer
mixer.init()
mixer.music.load("static/music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Streamlit UI elements
st.title("Driver Drowsiness Detection")
st.subheader("By Sree Harith C")

# Button to start/stop the webcam
if 'run' not in st.session_state:
    st.session_state.run = False

def toggle_video():
    st.session_state.run = not st.session_state.run

# Dynamic button text based on the state
button_text = "Start" if not st.session_state.run else "Stop"
st.button(button_text, on_click=toggle_video)

# Threshold values
thresh = 0.25
frame_check = 20

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Video capture
cap = None
flag = 0

# Function to process video frames
def process_frame():
    global flag
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        return None

    frame = cv2.resize(frame, (450, 450))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to get face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract the landmark positions for the eyes
            left_eye = [
                face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]
            ]
            right_eye = [
                face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]
            ]

            # Convert normalized landmarks to coordinates
            left_eye_coords = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in left_eye]
            right_eye_coords = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in right_eye]

            # Calculate EAR for both eyes
            leftEAR = eye_aspect_ratio(left_eye_coords)
            rightEAR = eye_aspect_ratio(right_eye_coords)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw the eye contours
            cv2.polylines(frame, [np.array(left_eye_coords, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_eye_coords, dtype=np.int32)], True, (0, 255, 0), 1)

            # Check if EAR is below the threshold
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0

    return frame

# Streamlit application loop
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Create an empty container for the video

    while st.session_state.run:
        frame = process_frame()
        if frame is not None:
            # Resize frame to a moderate size
            desired_width = 640  # Set the desired width for moderate size
            height, width, _ = frame.shape
            aspect_ratio = width / height
            desired_height = int(desired_width / aspect_ratio)
            frame = cv2.resize(frame, (desired_width, desired_height))
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the video frame with center alignment
            stframe.image(frame, channels="RGB", use_column_width=False, width=desired_width)
            time.sleep(0.1)  # Adjust to control the refresh rate

    cap.release()
    cv2.destroyAllWindows()
else:
    if cap is not None and cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
    st.write("Video is stopped. Click 'Start Video' to begin.")
