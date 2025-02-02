import cv2
import mediapipe as mp
import face_recognition

# Load the known face encoding (this is your reference face)
known_image = face_recognition.load_image_file("yu67.png")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize MediaPipe Face Detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize OpenCV's Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture (webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (required by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe face detection
    results = face_detection.process(rgb_frame)

    # Convert the frame to grayscale for OpenCV Haar Cascade face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw bounding boxes for faces detected by OpenCV Haar Cascade
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw bounding boxes for faces detected by MediaPipe
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Perform face recognition for faces detected in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]

            # Compare with the known encoding
            match = face_recognition.compare_faces([known_encoding], face_encoding)
            if match[0]:
                cv2.putText(frame, "Match!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Match", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the video feed with the bounding boxes and recognition results
    cv2.imshow('Video', frame)

    # Exit the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
