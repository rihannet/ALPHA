import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# Function to extract frames from a video file
def frames_from_video_file(video_path, n_frames, frame_size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame_list.append(frame)
    cap.release()
    
    for start in range(len(frame_list) - n_frames + 1):
        sequence = frame_list[start:start + n_frames]
        frames.append(np.array(sequence))
    return np.array(frames)

# Dataset class for loading frames
class FrameDataset(Dataset):
    def __init__(self, path, n_frames, training=False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(os.listdir(self.path))
        self.class_ids_for_name = {name: idx for idx, name in enumerate(self.class_names)}
        self.video_paths, self.classes = self.get_files_and_class_names()

    def get_files_and_class_names(self):
        video_paths = []
        classes = []
        for class_name in self.class_names:
            class_path = os.path.join(self.path, class_name)
            if os.path.isdir(class_path):
                for video_name in os.listdir(class_path):
                    video_paths.append(os.path.join(class_path, video_name))
                    classes.append(self.class_ids_for_name[class_name])
        return video_paths, classes

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.classes[idx]
        video_frames = frames_from_video_file(video_path, self.n_frames)
        if video_frames.shape[0] > 0:
            return torch.from_numpy(video_frames[0] / 255.0).permute(3, 0, 1, 2).float(), label

# Display a batch of frames
def display_batch(data_loader):
    for video_frames, labels in data_loader:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(video_frames[0].permute(1, 2, 0).numpy())
        ax.set_title(f'Label: {labels[0]}')
        ax.axis('off')
        plt.show()
        break

# Dataset parameters
data_directory = 'X:/stage1/data'  # Update with your dataset path
batch_size = 4
dataset = FrameDataset(data_directory, 16, training=True)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the 3D ResNet model
class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d((1, 2, 2))
        # Reduce the number of residual blocks and channels
        self.res_block1 = self.make_res_block(32)
        self.res_block2 = self.make_res_block(32)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, num_classes)

    def make_res_block(self, channels):
        layers = []
        # Use only two layers per residual block to make it simpler
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

# Initialize model, loss, optimizer
model = ResNet3D(num_classes=len(dataset.class_names))  # Adjust num_classes based on dataset
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    model = model.cuda()

# Training function
def train_model(model, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model(model, criterion, optimizer)

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for inputs, label in data_loader:
            if torch.cuda.is_available():
                inputs, label = inputs.cuda(), label.cuda()
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

evaluate(model, test_loader)

# Save model
torch.save(model.state_dict(), 'autism_model.pth')

# Load model
loaded_model = ResNet3D(num_classes=len(dataset.class_names))  # Adjust num_classes based on dataset
loaded_model.load_state_dict(torch.load('autism_model.pth'))
loaded_model.eval()
