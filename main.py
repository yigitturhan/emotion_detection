import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import time

model = load_model('fer_model.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.expand_dims(face_normalized, axis=0)
            face_reshaped = np.expand_dims(face_reshaped, axis=-1)
            prediction = model.predict(face_reshaped)
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
            break

        cv2.imshow('Webcam Video - Emotion Detection', frame)
    else:
        print("Error: Could not read frame.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
