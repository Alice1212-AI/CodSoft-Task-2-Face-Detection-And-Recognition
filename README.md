# CodSoft-Task-2-Face-Detection-And-Recognition
Real-Time Face Detection and Recognition

This project implements real-time face detection and recognition using OpenCV, MediaPipe, and face_recognition. It detects faces using Haar Cascade (OpenCV) and MediaPipe Face Detection, then compares detected faces with a known reference image for recognition.

# Features:

✔ Real-time face detection using MediaPipe and Haar Cascade

✔ Face recognition using face_recognition library

✔ Bounding boxes for detected faces with recognition results

✔ Works with live webcam feed

# Requirements:

Ensure you have Python installed, then install the required libraries:
pip install opencv-python mediapipe face-recognition numpy

# Usage:

1. Replace "yu67.png" with your reference image for face recognition.
 
2. Run the script:
python face_detection.py

4. The webcam feed will open, detecting faces and recognizing the reference image.
 
5. Press 'q' to exit.

# How It Works

Face Detection: Uses both MediaPipe and Haar Cascade for accurate face detection.

Face Recognition: Compares detected faces with the reference face encoding using face_recognition.

# Bounding Boxes:

Green: Haar Cascade detected faces

Red: MediaPipe detected faces

Match/No Match: Displayed for recognized faces

# Example Output

When a face is detected and recognized, it displays "Match!" or "No Match" over the detected face.
