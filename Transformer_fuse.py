import scipy.io
import librosa
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
import numpy as np
from scipy.io import wavfile
import random
from scipy.io import loadmat
import random
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE

def extract_sound_features(file_path):
    # Load audio file using librosa
    audio, sr = librosa.load(file_path, sr=None)

    # Extract MFCCs as features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    # Apply preprocessing to sound signal (e.g., denoising, silence removal)

    # Compute first-order and second-order differences
    delta1 = librosa.feature.delta(mfccs, order=1)
    delta2 = librosa.feature.delta(mfccs, order=2)

    # Concatenate MFCCs, first-order differences, and second-order differences
    features = np.concatenate((mfccs, delta1, delta2), axis=0)

    # Flatten the features
    return features.flatten()

def extract_vibration_features(file_path):
    # Load MATLAB file
    mat_data = loadmat(file_path)

    # Check the available keys in the loaded MATLAB data
    # print(mat_data.keys())

    # Replace 'vibration_data' with the correct key name based on the output of print(mat_data.keys())
    vibration_feature = mat_data['vib_data']

    # Normalize the vibration feature
    norm_vibration_feature = (vibration_feature - np.mean(vibration_feature)) / np.std(vibration_feature)

    return norm_vibration_feature

# Load training data
train_mat = glob.glob('./data/train/*/*/*.mat')
train_mat.sort()

train_wav = glob.glob('./data/train/*/*/*.wav')
train_wav.sort()

class_name_to_id = {'normal': 0, 'cage': 1, 'outer': 2, 'inner': 3, 'roller': 4}
train_sound_data = [extract_sound_features(path) for path in train_wav]
train_vibration_feat = [extract_vibration_features(path) for path in train_mat]
train_label = [class_name_to_id[path.split('_')[-4]] for path in train_mat]
# print("Train Class Labels:", train_label)
# Create StandardScaler objects for sound and vibration features
scaler_sound = StandardScaler()
scaler_vibration = StandardScaler()

# Convert lists to NumPy arrays
train_sound_data = np.array(train_sound_data)
train_vibration_feat = np.array(train_vibration_feat)

# Reshape the train_sound_data array to 2-dimensional
num_samples_train, num_features_train = train_sound_data.shape
train_sound_data = train_sound_data.reshape(num_samples_train, num_features_train)

# Standardize the features using the created scalers
train_sound_data = scaler_sound.fit_transform(train_sound_data)

# Reshape the train_vibration_feat array to 2-dimensional
num_samples_train_vib, num_features_train_vib, _ = train_vibration_feat.shape
train_vibration_feat = train_vibration_feat.reshape(num_samples_train_vib, num_features_train_vib)

# Standardize the features using the created scaler
train_vibration_feat = scaler_vibration.fit_transform(train_vibration_feat)

# Load validation data
val_mat = glob.glob('./data/val/*/*/*.mat')
val_mat.sort()

val_wav = glob.glob('./data/val/*/*/*.wav')
val_wav.sort()

val_vibration_feat = [extract_vibration_features(path) for path in val_mat]
val_sound_data = [extract_sound_features(path) for path in val_wav]
val_label = [class_name_to_id[path.split('_')[-4]] for path in val_mat]
# print("Validation Class Labels:", val_label)
# Convert lists to NumPy arrays
val_sound_data = np.array(val_sound_data)
val_vibration_feat = np.array(val_vibration_feat)

# Reshape the val_sound_data array to 2-dimensional
num_samples_val, num_features_val = val_sound_data.shape
val_sound_data = val_sound_data.reshape(num_samples_val, num_features_val)

# Standardize the features using the same scaler used for training data
val_sound_data = scaler_sound.transform(val_sound_data)

# Reshape the val_vibration_feat array to 2-dimensional
num_samples_val_vib, num_features_val_vib, _ = val_vibration_feat.shape
val_vibration_feat = val_vibration_feat.reshape(num_samples_val_vib, num_features_val_vib)

# Standardize the vibration features using the same scaler used for training data
val_vibration_feat = scaler_vibration.transform(val_vibration_feat)

class SoundVibrationDataset(data.Dataset):
    def __init__(self, sound_data, vibration_data, labels, sound_transform=None, vibration_transform=None):
        self.sound_data = sound_data
        self.vibration_data = vibration_data
        self.labels = labels
        self.sound_transform = sound_transform
        self.vibration_transform = vibration_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sound = self.sound_data[index]
        vibration = self.vibration_data[index]
        label = self.labels[index]

        if self.sound_transform:
            sound = self.sound_transform(sound)
        if self.vibration_transform:
            vibration = self.vibration_transform(vibration)

        return sound, vibration, label

# Data augmentation for sound data
class SoundDataAugmentation:
    def __init__(self, speed_change_factors=[0.9, 1.0, 1.1]):
        self.speed_change_factors = speed_change_factors

    def __call__(self, waveform):
        speed_change_factor = random.choice(self.speed_change_factors)
        # Convert to mono audio
        waveform = librosa.to_mono(waveform)
        # Apply time stretching
        waveform = librosa.effects.time_stretch(waveform, rate=speed_change_factor)
        # Add an extra dimension to make it 2D
        waveform = np.expand_dims(waveform, axis=0)
        return waveform


import torch
from torch.utils.data import Dataset
class VibrationData(Dataset):
    def __init__(self, sound_data, vibration_data, labels, sound_transform=None, vibration_transform=None):
        self.sound_data = sound_data
        self.vibration_data = vibration_data
        self.labels = labels
        self.sound_transform = sound_transform
        self.vibration_transform = vibration_transform

    def __getitem__(self, index):
        sound = self.sound_data[index]
        vibration = self.vibration_data[index]
        label = self.labels[index]

        if self.sound_transform is not None:
            sound = self.sound_transform(sound)

        if self.vibration_transform is not None:
            vibration = self.vibration_transform(vibration)

        return sound, vibration, label

    def __len__(self):
        return len(self.labels)

def collate_tensor_fn(batch):
    max_sound_length = max([sound.shape[-1] for sound, _, _ in batch])
    max_vibration_length = max([vibration.shape[-1] for _, vibration, _ in batch])

    # Pad the shorter sound and vibration tensors with zeros to match the maximum lengths
    padded_sounds = [torch.nn.functional.pad(torch.tensor(sound), (0, max_sound_length - sound.shape[-1])) for
                        sound, _, _ in batch]
    padded_vibrations = [
        torch.nn.functional.pad(torch.tensor(vibration), (0, max_vibration_length - vibration.shape[-1])) for
        _, vibration, _ in batch]
    stacked_sounds = torch.stack(padded_sounds)
    stacked_vibrations = torch.stack(padded_vibrations)

    # Convert labels to integers
    labels = torch.tensor([int(label) for _, _, label in batch])

    return stacked_sounds, stacked_vibrations, labels

# Data augmentation for vibration data
class VibrationDataAugmentation:
    def __init__(self, factor=0.2):
        self.factor = factor

    def __call__(self, vibration):
        if isinstance(vibration, np.ndarray):
            vibration = torch.tensor(vibration)  # Convert NumPy array to PyTorch tensor

        # Apply data augmentation to the vibration data
        augmented_vibration = vibration + self.factor * torch.randn_like(vibration)
        return augmented_vibration.numpy()  # Convert back to NumPy array if needed

# Instantiate the model
import torch.nn.functional as F
class SoundVibrationTransformer(nn.Module):
    def __init__(self, sound_input_dim, vibration_input_dim, num_classes):
        super(SoundVibrationTransformer, self).__init__()

        # Define the layers for sound data processing
        self.sound_fc1 = nn.Linear(sound_input_dim, 128)
        self.sound_fc2 = nn.Linear(128, 64)

        # Define the layers for vibration data processing
        self.vibration_fc1 = nn.Linear(vibration_input_dim, 128)
        self.vibration_fc2 = nn.Linear(128, 64)

        # Set the data type of weight parameters to float32
        self.sound_fc1.weight.data = self.sound_fc1.weight.data.to(torch.float32)
        self.sound_fc2.weight.data = self.sound_fc2.weight.data.to(torch.float32)
        self.vibration_fc1.weight.data = self.vibration_fc1.weight.data.to(torch.float32)
        self.vibration_fc2.weight.data = self.vibration_fc2.weight.data.to(torch.float32)

        # Define the fusion layer
        self.fusion_fc1 = nn.Linear(128, 64)
        self.fusion_fc2 = nn.Linear(64, num_classes)

        # Dropout layers
        self.sound_dropout = nn.Dropout(0.5)
        self.vibration_dropout = nn.Dropout(0.5)

        # Batch normalization layers
        self.sound_bn1 = nn.BatchNorm1d(128)
        self.vibration_bn1 = nn.BatchNorm1d(128)

    def forward(self, sound, vibration):
        # Sound data processing
        sound = sound.view(sound.size(0), -1)  # Reshape sound to (batch_size, 22560)
        sound = F.relu(self.sound_fc1(sound))
        sound = self.sound_bn1(sound)
        sound = self.sound_dropout(sound)

        sound = F.relu(self.sound_fc2(sound))
        sound = self.sound_dropout(sound)

        # Vibration data processing
        vibration = F.relu(self.vibration_fc1(vibration.to(torch.float32)))  # Convert vibration to float32
        vibration = self.vibration_bn1(vibration)
        vibration = self.vibration_dropout(vibration)

        vibration = F.relu(self.vibration_fc2(vibration))
        vibration = self.vibration_dropout(vibration)

        # Concatenate the processed sound and vibration features
        combined_features = torch.cat((sound, vibration), dim=1)

        # Fusion layer
        fused_features = F.relu(self.fusion_fc1(combined_features))
        outputs = self.fusion_fc2(fused_features)

        return outputs


# Hyperparameters
batch_size = 64
epochs = 100
num_classes = len(np.unique(train_label))
print("Number of classes in the dataset:", num_classes)
num_classes = 5
# Define data transformations
sound_transform = transforms.Compose([
    SoundDataAugmentation(),
    transforms.ToTensor(),
])

vibration_transform = VibrationDataAugmentation(factor=0.5)  # You can adjust the factor as needed

# Create PyTorch DataLoader
train_dataset = SoundVibrationDataset(train_sound_data, train_vibration_feat, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_tensor_fn)

val_dataset = SoundVibrationDataset(val_sound_data, val_vibration_feat, val_label)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_tensor_fn)

# Instantiate the model
model = SoundVibrationTransformer(
    sound_input_dim=train_sound_data.shape[1],  # Remove the multiplication by 3
    vibration_input_dim=train_vibration_feat.shape[1],
    num_classes=num_classes,
)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Initialize empty lists to store training and validation metrics and intermediate features
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
intermediate_features = []

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for sound, vibration, labels in train_loader:
        sound, vibration, labels = sound.to(device), vibration.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sound, vibration)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        correct_predictions += (predictions == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / len(train_dataset)

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0

    with torch.no_grad():
        for sound, vibration, labels in val_loader:
            sound, vibration, labels = sound.to(device), vibration.to(device), labels.to(device)

            outputs = model(sound, vibration)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            val_correct_predictions += (predictions == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct_predictions / len(val_dataset)

    # Append the metrics to the corresponding lists
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

    # Save intermediate features every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        intermediate_output = []

        with torch.no_grad():
            for sound, vibration, _ in train_loader:
                sound, vibration = sound.to(device), vibration.to(device)
                outputs = model(sound, vibration)
                intermediate_output.extend(outputs.cpu().numpy())

        intermediate_features.append(intermediate_output)

# Create a DataFrame for the training and validation results
results_df = pd.DataFrame({
    'Epoch': range(1, epochs+1),
    'Train Loss': train_losses,
    'Train Accuracy': train_accuracies,
    'Val Loss': val_losses,
    'Val Accuracy': val_accuracies
})

# Save the training and validation results to a CSV file
results_df.to_csv('training_validation_results.csv', index=False)
from sklearn.manifold import TSNE
class_features = tsne_features[np.isin(indices, class_indices)]
# Visualize intermediate features using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300, learning_rate=200)

epochs_to_visualize = [0, 20, 40, 60, 80, 100]

for epoch in epochs_to_visualize:
    # Get the intermediate output for the current epoch
    intermediate_output = intermediate_features[epoch // 20]  # Use correct intermediate features

    # Randomly sample a subset of the data for t-SNE visualization
    num_samples_to_visualize = 1000
    if len(intermediate_output) > num_samples_to_visualize:
        indices = np.random.choice(len(intermediate_output), num_samples_to_visualize, replace=False)
        intermediate_output = [intermediate_output[i] for i in indices]  # Use correct variable name

    tsne_features = tsne.fit_transform(intermediate_output)

    # Create a color map for class labels
    class_color_map = plt.cm.tab10

    plt.figure(figsize=(8, 6))
    for class_id in range(len(class_name_to_id)):
        # Get indices of samples belonging to the current class
        class_indices = np.where(np.array(val_label) == class_id)[0]
        # Get t-SNE features of samples for the current class
        class_features = tsne_features[np.isin(indices, class_indices)]

        # Scatter plot for samples of the current class
        plt.scatter(class_features[:, 0], class_features[:, 1], c=[class_color_map(class_id)], label=list(class_name_to_id.keys())[class_id], s=10)

    plt.title(f"t-SNE Visualization - Epoch {epoch}")
    plt.legend()
    plt.show()
# Plot training and validation loss curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Plot training and validation accuracy curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()
plt.show()
# Save the model
torch.save(model.state_dict(), 'best_model.pt')

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

# Evaluation on validation set
val_predictions_list = []
model.eval()

with torch.no_grad():
    for sound, vibration, _ in val_loader:
        sound, vibration = sound.to(device), vibration.to(device)
        outputs = model(sound, vibration)
        _, predictions = torch.max(outputs, 1)
        val_predictions_list.append(predictions.cpu().numpy())
# Model Fusion using majority voting with padding and default label
def majority_voting(predictions_list, num_samples_val, default_label):
    final_predictions = []
    for i in range(num_samples_val):
        predictions_at_i = [predictions[i] for predictions in predictions_list if len(predictions) > i]
        if not predictions_at_i:
            final_predictions.append(default_label)
        else:
            count = Counter(predictions_at_i)
            most_common_label = count.most_common(1)[0][0]
            final_predictions.append(most_common_label)
    return final_predictions

# Calculate the number of samples in the validation set
num_samples_val = len(val_label)

# Define the default label (choose the label with the highest occurrence in the training set)
default_label = max(set(train_label), key=train_label.count)

# Perform predictions on the validation set using the trained model
val_predictions_list = []
model.eval()

with torch.no_grad():
    for sound, vibration, _ in val_loader:
        sound, vibration = sound.to(device), vibration.to(device)
        outputs = model(sound, vibration)
        _, predictions = torch.max(outputs, 1)
        val_predictions_list.append(predictions.cpu().numpy())

# Define the default label (choose the label with the highest occurrence in the training set)
default_label = max(set(train_label), key=train_label.count)

# Calculate the number of samples in the validation set
num_samples_val = len(val_label)

# Model Fusion using majority voting with padding and default label
final_val_predictions = majority_voting(val_predictions_list, num_samples_val, default_label)

# Calculate the validation accuracy
val_accuracy = accuracy_score(val_label, final_val_predictions)

val_cm = confusion_matrix(val_label, final_val_predictions)
val_classification_report = classification_report(val_label, final_val_predictions)
import pandas as pd

# Create DataFrames for val_label and final_val_predictions
val_label_df = pd.DataFrame({'True_Label': val_label})
final_val_predictions_df = pd.DataFrame({'Predicted_Label': final_val_predictions})

# Save DataFrames to CSV files
val_label_df.to_csv('val_label.csv', index=False)
final_val_predictions_df.to_csv('final_val_predictions.csv', index=False)

import seaborn as sns

# Create the confusion matrix
cm = confusion_matrix(val_label, final_val_predictions)

# Create a DataFrame for the confusion matrix
class_names = list(class_name_to_id.keys())
cm_df = pd.DataFrame(0, index=class_names, columns=class_names)

# Update the confusion matrix DataFrame with the actual values
for i in range(len(val_label)):
    true_label = class_names[val_label[i]]
    predicted_label = class_names[final_val_predictions[i]]
    cm_df.loc[true_label, predicted_label] += 1

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Validation Set')
plt.show()

print("Validation Accuracy: {:.4f}".format(val_accuracy))
print("Confusion Matrix:")
print(val_cm)
print("Classification Report:")
print(val_classification_report)

# Load test data
test_mat = glob.glob('./data/test/*.mat')
test_mat.sort()

test_wav = glob.glob('./data/test/*.wav')
test_wav.sort()

test_vibration_feat = [extract_vibration_features(path) for path in test_mat]
test_sound_data = [extract_sound_features(path) for path in test_wav]

# Convert lists to NumPy arrays
test_sound_data = np.array(test_sound_data)
test_vibration_feat = np.array(test_vibration_feat)

# Standardize the features using the same scalers used for training data
test_sound_data = scaler_sound.transform(test_sound_data)

# Reshape the test_vibration_feat array to 2-dimensional
num_samples_test_vib = len(test_vibration_feat)
num_features_test_vib = test_vibration_feat[0].shape[0]
test_vibration_feat = test_vibration_feat.reshape(num_samples_test_vib, num_features_test_vib)

# Standardize the vibration features using the same scaler used for training data
test_vibration_feat = scaler_vibration.transform(test_vibration_feat)

# Create PyTorch DataLoader for the test set
test_dataset = SoundVibrationDataset(test_sound_data, test_vibration_feat, None)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

# Perform predictions on the test set using the trained model
test_predictions_list = []
model.eval()

with torch.no_grad():
    for sound, vibration, _ in test_loader:
        sound, vibration = sound.to(device), vibration.to(device)
        outputs = model(sound, vibration)
        _, predictions = torch.max(outputs, 1)
        test_predictions_list.append(predictions.cpu().numpy())

# Model Fusion using majority voting
final_test_predictions = majority_voting(test_predictions_list)

# Generate submission file
submission_df = pd.DataFrame({'audio_name': [x.split('/')[-1] for x in test_wav], 'label': final_test_predictions})
submission_df.to_csv('submit.csv', index=None)
