import os
import glob
import scipy.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

class FaultDiagnosisModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FaultDiagnosisModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        conv_output_size = self._calculate_conv_output_size(input_dim)
        self.lstm = nn.LSTM(conv_output_size, 64, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(128, num_heads=4)
        self.fc = nn.Linear(128, num_classes)

    def _calculate_conv_output_size(self, input_dim):
        with torch.no_grad():
            sample_data = torch.randn(1, 1, input_dim)
            conv_output = self.conv(sample_data)
        return conv_output.size(2)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), -1, x.size(-1))
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

from sklearn.preprocessing import LabelEncoder
class FaultDataset(Dataset):
    def __init__(self, mat_paths, labels):
        self.mat_paths = mat_paths
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.mat_paths)

    def __getitem__(self, idx):
        mat_data = scipy.io.loadmat(self.mat_paths[idx])['vib_data']
        label = self.labels[idx]
        mat_data = torch.tensor(mat_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        label = torch.tensor(label, dtype=torch.long)
        return mat_data, label


# Set the random seed for reproducibility
torch.manual_seed(42)

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

train_label = [os.path.basename(os.path.dirname(path)) for path in train_mat]
val_label = [os.path.basename(os.path.dirname(path)) for path in val_mat]
test_label = [os.path.basename(os.path.splitext(path)[0]) for path in test_wav]

train_dataset = FaultDataset(train_mat, train_label)
val_dataset = FaultDataset(val_mat, val_label)
print("Number of training files:", len(train_mat))
print("Number of samples in train_dataset:", len(train_dataset))
sample = train_dataset[0]  # Get the first sample
print("Sample shape:", sample[0].shape)  # Input data shape
print("Sample label:", sample[1])  # Label

# Set up data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and optimizer
input_dim = sample[0].shape[1]  # Update input_dim to match the sequence length
num_classes = len(np.unique(train_label))
model = FaultDiagnosisModel(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
sample_data = torch.randn(1, 1, 20480)  # Create a sample input with batch size 1 and 1 channel
sample_data = sample_data.to(device)
sample_output = model(sample_data)
print("Sample output shape:", sample_output.shape)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation on validation set
    model.eval()
    val_predictions = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_predictions.extend(predicted.cpu().numpy())

    val_accuracy = accuracy_score(val_label, val_predictions)
    val_cm = confusion_matrix(val_label, val_predictions)
    classification_report_val = classification_report(val_label, val_predictions, zero_division=1)

    print("Epoch:", epoch + 1)
    print("Accuracy (Validation Set):", val_accuracy)
    print("Confusion Matrix (Validation Set):\n", val_cm)
    print("Classification Report (Validation Set):\n", classification_report_val)
    print()

# Load and process test data
test_dataset = FaultDataset(test_mat, test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Test the model on the test set
model.eval()
test_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().numpy())

# Print classification report and other evaluation metrics for the test set
test_accuracy = accuracy_score(test_label, test_predictions)
test_cm = confusion_matrix(test_label, test_predictions)
classification_report_test = classification_report(test_label, test_predictions, zero_division=1)

print("Accuracy (Test Set):", test_accuracy)
print("Confusion Matrix (Test Set):\n", test_cm)
print("Classification Report (Test Set):\n", classification_report_test)

# Generate submission file
submission_df = pd.DataFrame({'audio_name': [os.path.basename(path) for path in test_wav], 'label': test_predictions})
submission_df.to_csv('submit.csv', index=None)
