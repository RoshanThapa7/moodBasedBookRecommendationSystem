import mediapipe as mp
import numpy as np
import cv2
import os

name = input("Enter the name of the data: ")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

# Check if the file already exists BEFORE initializing the webcam
if os.path.exists(f"{name}.npy"):
    try:
        X = list(np.load(f"{name}.npy"))
        data_size = len(X)
        print(f"Loaded {data_size} existing samples from {name}.npy")
    except Exception as e:
        print(f"Error loading existing data: {e}")
        X = [] # Ensure X is initialized even if loading fails
        data_size = 0

# Initialize webcam AFTER loading existing data
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit() # Exit if webcam cannot be opened

new_samples = 0
try:
    while True:
        lst = []
        _, frm = cap.read()

        if not _:
            print("Error: Could not read frame from webcam")
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            X.append(lst)
            data_size = data_size + 1
            new_samples += 1

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("window", frm)

        if cv2.waitKey(1) == 27 or new_samples >= 100:
            break

except Exception as e:
    print(f"An error occurred during data collection: {e}")

finally:
    # Ensure proper release of resources even if an error occurs
    cv2.destroyAllWindows()
    cap.release()

    # Option to delete last 100 samples
    delete_option = input("Delete last 100 samples? (y/n): ")
    if delete_option.lower() == 'y':
        if len(X) >= 100:
            X = X[:-100]
            data_size = len(X)
            print("Last 100 samples deleted.")
        else:
            print("Not enough samples to delete (less than 100).")

    # Save the updated data by appending to the existing file (or creating a new one if it's the first run)
    try:
        np.save(f"{name}.npy", np.array(X))
        print(f"Saved {len(X)} samples to {name}.npy")
        print(np.array(X).shape)
    except Exception as e:
        print(f"Error saving data to file: {e}")