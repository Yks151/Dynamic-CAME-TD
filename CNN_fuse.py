import scipy.io
import librosa
import numpy as np
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to extract vibration features
def extract_vibration_features(file_path):
    mat_data = scipy.io.loadmat(file_path)
    vibration_data = mat_data['vib_data']
    # Add your vibration feature extraction code here
    # Extract statistical features such as mean, standard deviation, and maximum value
    mean = np.mean(vibration_data)
    std = np.std(vibration_data)
    maximum = np.max(vibration_data)

    # Concatenate the features into a feature vector
    features = np.array([mean, std, maximum])

    # Convert features to tensor
    features_tensor = torch.tensor(features)

    return features_tensor


# Function to extract sound features
def extract_sound_features(file_path):
    signal, sr = librosa.load(file_path)

    # Extract MFCCs as features
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)

    # Apply preprocessing to sound signal (e.g., denoising, silence removal)

    # Compute first-order and second-order differences
    delta1 = librosa.feature.delta(mfccs, order=1)
    delta2 = librosa.feature.delta(mfccs, order=2)

    # Concatenate MFCCs, first-order differences, and second-order differences
    features = np.concatenate((mfccs, delta1, delta2), axis=0)

    return features.flatten()


# Custom dataset class
class FaultDataset(Dataset):
    def __init__(self, mat_files, wav_files, vibration_transform=None, sound_transform=None):
        self.mat_files = mat_files
        self.wav_files = wav_files
        self.vibration_transform = vibration_transform
        self.sound_transform = sound_transform

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_path = self.mat_files[idx]
        wav_path = self.wav_files[idx]
        vibration_feat = extract_vibration_features(mat_path)
        sound_feat = extract_sound_features(wav_path)
        label = mat_path.split('_')[-4]

        # Modify the shape of the vibration feature tensor
        vibration_feat = vibration_feat.view(1, -1, 1, 1)

        if self.vibration_transform:
            vibration_feat = self.vibration_transform(vibration_feat)

        if self.sound_transform:
            sound_feat = self.sound_transform(sound_feat)

        return vibration_feat, sound_feat, label

# Load training data
train_mat = glob.glob('./data/train/*/*/*.mat')
train_mat.sort()

train_wav = glob.glob('./data/train/*/*/*.wav')
train_wav.sort()

# Load validation data
val_mat = glob.glob('./data/val/*/*/*.mat')
val_mat.sort()

val_wav = glob.glob('./data/val/*/*/*.wav')
val_wav.sort()

# Load test data
test_mat = glob.glob('./data/test/*.mat')
test_mat.sort()

test_wav = glob.glob('./data/test/*.wav')
test_wav.sort()

# Define data transformations for standardization
scaler = StandardScaler()


# Function to standardize features
def standardize_features(features):
    scaler.fit(features)
    standardized_features = scaler.transform(features)
    return standardized_features


# Extract features
train_vibration_feat = [extract_vibration_features(path) for path in train_mat]
train_sound_feat = [extract_sound_features(path) for path in train_wav]
train_label = [path.split('_')[-4] for path in train_mat]

val_vibration_feat = [extract_vibration_features(path) for path in val_mat]
val_sound_feat = [extract_sound_features(path) for path in val_wav]
val_label = [path.split('_')[-4] for path in val_mat]

test_vibration_feat = [extract_vibration_features(path) for path in test_mat]
test_sound_feat = [extract_sound_features(path) for path in test_wav]

# Standardize the features
train_vibration_feat = standardize_features(np.array(train_vibration_feat).reshape(-1, 1))
train_sound_feat = standardize_features(train_sound_feat)
val_vibration_feat = standardize_features(val_vibration_feat)
val_sound_feat = standardize_features(val_sound_feat)
test_vibration_feat = standardize_features(test_vibration_feat)
test_sound_feat = standardize_features(test_sound_feat)

# Convert to numpy arrays
train_vibration_feat = np.array(train_vibration_feat)
train_sound_feat = np.array(train_sound_feat)
val_sound_feat = np.array(val_sound_feat)
test_vibration_feat = np.array(test_vibration_feat)
test_sound_feat = np.array(test_sound_feat)

# Convert labels to numpy arrays
train_label = np.array(train_label)
val_label = np.array(val_label)

# Define CNN model
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)  # Modify the input channels to 32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_shape[0] // 3) * (input_shape[1] // 4), 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, vibration_feat, sound_feat):
        print("Input shape (vibration_feat):", vibration_feat.shape)
        print("Input shape (sound_feat):", sound_feat.shape)
        x = vibration_feat.float()
        print("After unsqueeze:", x.shape)
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        print("After conv2:", x.shape)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Get input shape
input_shape = (train_vibration_feat.shape[1], train_sound_feat.shape[1])

# Create train dataset
train_dataset = FaultDataset(train_mat, train_wav)

# Create validation dataset
val_dataset = FaultDataset(val_mat, val_wav)

# Create test dataset
test_dataset = FaultDataset(test_mat, test_wav)

# Define data loaders
batch_size = 32
learning_rate = 0.001
num_epochs = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define CNN model
model = CNN(input_shape, num_classes=len(np.unique(train_label)))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for vibration_feat, sound_feat, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        print("Before forward pass")
        outputs = model(vibration_feat, sound_feat)
        print("After forward pass")
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track training loss and accuracy
        train_loss += loss.item() * vibration_feat.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    train_loss /= len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for vibration_feat, sound_feat, labels in val_loader:
            # Forward pass
            outputs = model(vibration_feat, sound_feat)
            loss = criterion(outputs, labels)

            # Track validation loss and accuracy
            val_loss += loss.item() * vibration_feat.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    val_loss /= len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)

    # Print training progress
    print(f'Epoch {epoch + 1}/{num_epochs}: '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Evaluation on test set
model.eval()
test_predictions = []

with torch.no_grad():
    for vibration_feat, sound_feat, _ in test_loader:
        # Forward pass
        outputs = model(vibration_feat, sound_feat)

        # Get predicted labels
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.tolist())

# Generate submission file
submission_df = pd.DataFrame({'audio_name': [x.split('/')[-1] for x in test_wav], 'label': test_predictions})
submission_df.to_csv('submit.csv', index=None)

# Print classification report and other evaluation metrics for the test set
test_label = [path.split('_')[-1].split('.')[0] for path in test_wav]
test_accuracy = accuracy_score(test_label, test_predictions)
test_cm = confusion_matrix(test_label, test_predictions)
classification_report_test = classification_report(test_label, test_predictions, zero_division=1)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Confusion Matrix:\n{test_cm}')
print(f'Classification Report:\n{classification_report_test}')
