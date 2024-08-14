import scipy.io
import librosa
import librosa.display
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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

    return features.flatten()

# Modify the extract_vibration_features function to return exactly 20 features
def extract_vibration_features(file_path):
    mat_data = scipy.io.loadmat(file_path)
    vibration_data = mat_data['vib_data']

    # Apply Continuous Fourier Transform to obtain the frequency domain representation
    freq_domain = np.fft.fft(vibration_data, axis=0)
    amplitude = np.abs(freq_domain)

    # Extract statistical features such as mean, standard deviation, and maximum value
    mean = np.mean(amplitude, axis=0)
    std = np.std(amplitude, axis=0)
    maximum = np.max(amplitude, axis=0)

    # Concatenate the features into a feature vector
    features = np.hstack((mean, std, maximum))

    # Keep only the first 20 features
    num_features = 20
    if features.shape[0] > num_features:
        features = features[:num_features]

    return features

# Set the folder path
folder_path = './data/val/normal/'

# Get all file paths for wav and mat files in the folder
wav_files = glob.glob(folder_path + '*.wav')
mat_files = glob.glob(folder_path + '*.mat')

# Sort the file paths to ensure they are in the same order
wav_files.sort()
mat_files.sort()

# Initialize empty lists to store features
all_sound_features = []
all_vibration_features = []

# Iterate through each file and extract features
for wav_file, mat_file in zip(wav_files, mat_files):
    sound_features = extract_sound_features(wav_file)
    vibration_features = extract_vibration_features(mat_file)
    all_sound_features.append(sound_features)
    all_vibration_features.append(vibration_features)

# Convert lists to NumPy arrays
all_sound_features = np.array(all_sound_features)
all_vibration_features = np.array(all_vibration_features)

# Concatenate sound and vibration features into a single feature matrix
combined_features = np.hstack((all_sound_features, all_vibration_features))

# Reshape combined_features to a 2D array
combined_features = combined_features.reshape(-1, 40)

# Visualize and analyze the combined_features matrix
# Plot scatter plot of sound and vibration features
plt.figure(figsize=(8, 6))
plt.scatter(combined_features[:, :20], combined_features[:, 20:], s=30)
plt.title('Scatter Plot of Sound and Vibration Features')
plt.xlabel('Sound Features')
plt.ylabel('Vibration Features')
plt.grid(True)
plt.show()

# Calculate correlation matrix between sound and vibration features
correlation_matrix = np.corrcoef(combined_features[:, :20].T, combined_features[:, 20:].T)

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix between Sound and Vibration Features')
plt.xticks(np.arange(40), ['Sound' + str(i + 1) for i in range(20)] +
           ['Vibration' + str(i + 1) for i in range(20)], rotation=45)
plt.yticks(np.arange(40), ['Sound' + str(i + 1) for i in range(20)] +
           ['Vibration' + str(i + 1) for i in range(20)])
plt.show()

# Perform PCA on the combined features
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(combined_features)

# Plot 2D scatter plot after PCA
plt.figure(figsize=(8, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=30)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Scatter Plot of Sound and Vibration Features after PCA')
plt.grid(True)
plt.show()

# Plot Kernel Density Estimate Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(np.squeeze(combined_features[:, :20]), color='b', label='Sound')
sns.kdeplot(np.squeeze(combined_features[:, 20:]), color='r', label='Vibration')
plt.xlabel('Feature')
plt.ylabel('Density')
plt.title('Kernel Density Estimate Plot of Sound and Vibration Features')
plt.legend()
plt.show()

# Display the Spectrogram for sound and vibration data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(sample_sound_data))), sr=sr, y_axis='linear', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Sound Spectrogram')

plt.subplot(1, 2, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(sample_vibration_data))), sr=sr, y_axis='linear', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Vibration Spectrogram')

plt.tight_layout()
plt.show()