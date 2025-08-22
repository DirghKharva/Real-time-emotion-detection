import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque, Counter

# ------------------------------
# Load your trained CNN model
# ------------------------------
model = tf.keras.models.load_model('emotion_cnn.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# ------------------------------
# Load emojis
# ------------------------------
emoji_path = os.path.join(os.getcwd(), 'emojis')
emoji_dict = {}
for emotion in emotion_labels:
    file_path = os.path.join(emoji_path, f"{emotion.lower()}.png")
    if os.path.exists(file_path):
        emoji_dict[emotion] = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        print(f"[Warning] Emoji file missing: {file_path}")

# ------------------------------
# Initialize face detector
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------------------
# Initialize webcam
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise Exception("Camera not accessible! Check macOS permissions.")

# ------------------------------
# Emotion smoothing buffer
# ------------------------------
buffer_size = 10
emotion_buffer = deque(maxlen=buffer_size)

# ------------------------------
# Real-time loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        roi = roi_gray.reshape(1,48,48,1)/255.0

        # Predict emotion
        prediction = model.predict(roi, verbose=0)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_labels[maxindex]
        confidence = float(np.max(prediction))*100

        # Add to buffer for smoothing
        emotion_buffer.append(emotion)
        # Most common emotion in buffer
        emotion_to_show = Counter(emotion_buffer).most_common(1)[0][0]

        # ------------------------------
        # Draw rectangle and label
        # ------------------------------
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"{emotion_to_show} {confidence:.1f}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # ------------------------------
        # Overlay emoji above face
        # ------------------------------
        if emotion_to_show in emoji_dict:
            emoji = cv2.resize(emoji_dict[emotion_to_show], (w, w))
            y_offset = max(0, y - w)
            x_offset = x
            if emoji.shape[2] == 4:
                alpha_s = emoji[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    frame[y_offset:y_offset+w, x_offset:x_offset+w, c] = (
                        alpha_s * emoji[:, :, c] +
                        alpha_l * frame[y_offset:y_offset+w, x_offset:x_offset+w, c]
                    )

    cv2.imshow("Real-Time Emotion Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
