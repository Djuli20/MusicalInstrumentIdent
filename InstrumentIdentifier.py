import numpy as np
import torch
from torch import nn
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define the same AudioUtil class for processing
class AudioUtil:
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        spec = transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB()(spec)
        return spec

# Function to predict a single audio file
def predict_audio_category(audio_path, model, class_names):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load and preprocess the audio file
    aud = AudioUtil.open(audio_path)
    mel_spec = AudioUtil.spectro_gram(aud)

    # Ensure it has the right shape (1, 1, H, W)
    mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
    mel_spec = mel_spec.to(device) # Move mel_spec to the correct device

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(mel_spec)
        _, predicted = torch.max(output, 1)

    # Get the predicted class
    predicted_class = class_names[predicted.item()]
    
    return predicted_class

# Define model and load weights
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 21, 512)#self.fc1 = nn.Linear(128 * 4 * 29, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
model = SpectrogramCNN(num_classes=3).to(device)
model.load_state_dict(torch.load('modele/instrumente_trei.pth', map_location=device))
model.eval()

# Define class names
class_names = ["acordeon", "chitaraacustica", "pian"]  # Make sure these match your training labels

# Test with a single audio file
audio_file = "inregistraritrim/acordeon/20.wav_segment_3.wav"  # Replace with your actual file path
predicted_category = predict_audio_category(audio_file, model, class_names)

print(f"Predicted Instrument: {predicted_category}")