import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Define the autism class labels
autism_classes = {
    0: 'Asperger’s Syndrome',
    1: 'Classic Autism',
    2: 'PDD-NOS (Pervasive Developmental Disorder)',
    3: 'Rett Syndrome',
}

# Define the 3D ResNet model architecture
class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.res_block1 = self.make_res_block(32)
        self.res_block2 = self.make_res_block(32)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, num_classes)

    def make_res_block(self, channels):
        layers = []
        for _ in range(2):
            layers.append(nn.Conv3d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.relu(x))
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = ResNet3D(num_classes=len(autism_classes))  # Adjust based on your model definition
model.load_state_dict(torch.load('autism_model.pth'))  # Update with the path to your model
model.to(device)
model.eval()

# Function to read and preprocess video
def preprocess_video(video_path, n_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    frames_normalized = frames / 255.0  # Normalize to [0, 1]
    return frames_normalized

# Load and preprocess the video
video_path = 'X:/stage1/data/Rett Syndrome/WhatsApp Video 2024-11-06 at 10.16.43 PM.mp4'  # Update with your video path
video_frames = preprocess_video(video_path)
video_tensor = torch.tensor(video_frames, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)

# Perform prediction
with torch.no_grad():
    output = model(video_tensor)
    yhat = nn.Softmax(dim=1)(output)  # Apply softmax to get probabilities
    predicted_class = torch.argmax(yhat, dim=1).item()

# Display the result
predicted_label = autism_classes.get(predicted_class, "Unknown class")
print(f'Predicted class is: {predicted_label}')

# Display probabilities for each class
print("Model probabilities:")
for idx, prob in enumerate(yhat[0].cpu().numpy()):
    print(f"Probability of '{autism_classes[idx]}': {prob:.4f}")
