# face_recognition.py
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import datetime

# Load face classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('face_recognition_model.h5')

with open('dataset/labels.txt', 'r') as file:
    classes = file.read().split('\n')

# Open webcam
webcam = cv2.VideoCapture(0)

# Create or open file to save recognized data
output_file = 'recognized_faces.txt'
with open(output_file, 'a') as f:
    f.write("Recognized students\n")

while webcam.isOpened():
    status, frame = webcam.read()
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_crop = frame[y:y + h, x:x + w]

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]
        index = np.argmax(conf)
        label = classes[index]

        if conf[index] < 0.7:
            label = 'Unknown'

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Save data to file when 'p' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('p'):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_file, 'a') as f:
                f.write(f"{timestamp}, {label}\n")
            print(f"Data saved: {timestamp}, {label}")

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()