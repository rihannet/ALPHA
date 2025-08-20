import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to read and preprocess video
def preprocess_video(video_path, n_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Load and preprocess the video
video_path = '/kaggle/input/sad1111/sad.mp4'  # Update with your video path
video_frames = preprocess_video(video_path)

# Normalize and add a batch dimension
video_frames_normalized = video_frames / 255.0

# Load the saved model
model = load_model('/kaggle/input/test-model1-not-perfect/tensorflow2/default/1/resnet3d_model.h5')

# Perform predictions
yhat = model.predict(np.expand_dims(video_frames_normalized, axis=0))

# Assuming your model outputs probabilities for each class
predicted_class = np.argmax(yhat, axis=1)

# Display the result
if predicted_class == 0:
    print('Predicted class is Angry')
elif predicted_class == 1:
    print('Predicted class is Happy')
else:
    print('Predicted class is Sad')

# Optionally display probabilities
print(f"Model probabilities: {yhat}")
