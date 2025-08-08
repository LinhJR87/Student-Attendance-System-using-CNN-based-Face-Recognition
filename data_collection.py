import cv2
import os

# Initialize camera
cam = cv2.VideoCapture(0)
# Load face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Prompt for student ID and name separately
    student_id = input('\nEnter Student ID ==> ').strip()
    student_name = input('Enter Student Name ==> ').strip()
    face_name = f"{student_id}-{student_name}"
    file_path = 'dataset/labels.txt'

    # Read existing student IDs from labels.txt if it exists
    existing_ids = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            # Only take the first 8 characters (Student ID) from each line
            existing_ids = {line.split('-')[0] for line in file if line.strip()}

    # Check if Student ID already exists
    if student_id in existing_ids:
        print('\n Data for this student ID already exists')
        continue  # Ask for input again

    # Ensure dataset folder exists
    os.makedirs('dataset', exist_ok=True)
    # Ensure student folder exists
    student_folder = f'dataset/{face_name}'
    os.makedirs(student_folder, exist_ok=True)
    break

print("\n [NOTICE] Initializing face capture. Please look at the camera and wait...")

# Counter for number of face images
count = 0

# Labeling in the file
with open(file_path, 'a', encoding='utf-8') as file:
    file.write('\n' + face_name)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        gray_face = gray[y:y + h, x:x + w]

        # Only save if the detected face region is large enough
        if gray_face.shape[0] < 10 or gray_face.shape[1] < 10:
            continue

        try:
            cv2.imwrite(f"dataset/{face_name}/face_{face_name}_{count}.jpg", gray_face)
        except Exception as e:
            print(f"Error saving image: {e}")
            continue

        count += 1

        cv2.putText(img, f"Number of images: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Image', img)

    if cv2.waitKey(20) & 0xff == 27 or count >= 500:  # Press ESC or reach 500 images
        break

print("\n [NOTICE] Data collection process is complete!")
cam.release()
cv2.destroyAllWindows()