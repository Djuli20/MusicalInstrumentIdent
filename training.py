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

class AudioUtil():
    @staticmethod
    def open(audio_file):
        """Loads an audio file as a tensor."""
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        """Converts an audio signal to a Mel Spectrogram."""
        sig, sr = aud
        spec = transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB()(spec)  # Convert to dB scale
        return spec

class AudioDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """Loads dataset from a CSV file."""
        self.dataframe = pd.read_csv(csv_path)
        self.class_names = sorted(self.dataframe['Tip'].unique())
        self.class_to_index = {genre: i for i, genre in enumerate(self.class_names)}
        self.file_list = [(row['Path'], self.class_to_index[row['Tip']]) for _, row in self.dataframe.iterrows()]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_file, class_id = self.file_list[idx]
        aud = AudioUtil.open(audio_file)  # Load audio
        mel_spec = AudioUtil.spectro_gram(aud)  # Convert to Mel Spectrogram
        return mel_spec, class_id

def create_data_loader(csv_path, batch_size=16, shuffle=True):
    """Creates a PyTorch DataLoader from the dataset."""
    dataset = AudioDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Load the dummy CSV file
instrumente_df = pd.read_csv('audio_data.csv')

# Now you can use the code with 'audio_data.csv'
train_df, test_df = train_test_split(instrumente_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

class_names = instrumente_df['Tip'].unique()
print(class_names)

# Create train, validation, and test CSV files
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

# Create DataLoaders
train_loader = create_data_loader('train.csv', batch_size=16)
val_loader = create_data_loader('val.csv', batch_size=16)
test_loader = create_data_loader('test.csv', batch_size=16)

# Example of iterating through the train_loader:
for batch in train_loader:
    mel_spectrograms, labels = batch
    print("Batch Mel Spectrograms shape:", mel_spectrograms.shape)
    print("Batch Labels shape:", labels.shape)
    break


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 150
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_iterator = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for batch_idx, (inputs, labels) in enumerate(train_iterator):
        inputs = inputs.float().to(device)  
        labels = labels.to(device) 
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        train_iterator.set_postfix(loss=running_loss / ((batch_idx + 1) * train_loader.batch_size))  # Update progress bar

    epoch_loss = running_loss / len(train_loader.dataset)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'modele/instrumente_trei.pth')

print(f"Best validation accuracy: {best_accuracy:.2%}")