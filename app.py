import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from scipy.spatial import distance
import dlib
from imutils import face_utils
import pygame

# Initialize Pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("static/music.wav")

# Define function for calculating Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for drowsiness detection
thresh = 0.25
frame_check = 20

# Initialize dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Define indexes for accessing left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Video Transformer for Streamlit WebRTC
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.flag = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Apply white balance correction
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(result)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

        rects = detect(gray, 0)

        for rect in rects:
            shape = predict(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(final, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(final, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                self.flag += 1
                if self.flag >= frame_check:
                    cv2.putText(final, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    pygame.mixer.music.play()
            else:
                self.flag = 0

        return final

# Streamlit App
st.title("Drowsiness Detection")
st.write("This app detects drowsiness using the Eye Aspect Ratio (EAR) method.")

# Set high resolution for video stream
webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
            "facingMode": "user"  # Use "user" for front camera on mobile devices
        },
        "audio": False,
    }
)
