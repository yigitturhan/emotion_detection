import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import time

# Loading of the pre-trained model
model = load_model('fer_model.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If the frame was captured successfully
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            # Extract the face from the frame
            face = gray[y:y+h, x:x+w]
            # Resize the face to 48x48 pixels
            face_resized = cv2.resize(face, (48, 48))
            # Normalize the face image
            face_normalized = face_resized / 255.0
            # Expand dimensions to match the input shape of the model (1, 48, 48, 1)
            face_reshaped = np.expand_dims(face_normalized, axis=0)
            face_reshaped = np.expand_dims(face_reshaped, axis=-1)
            # Predict the emotion
            prediction = model.predict(face_reshaped)
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]
            # Draw the rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Put the emotion label above the rectangle
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
            break

        # Display the resulting frame
        cv2.imshow('Webcam Video - Emotion Detection', frame)
    else:
        print("Error: Could not read frame.")
        break

    # Press 'q' on the keyboard to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
