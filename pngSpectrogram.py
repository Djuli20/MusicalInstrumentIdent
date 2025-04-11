import os
import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
from torchaudio import transforms

# Load dataset
csv_path = "audio_data.csv"
save_dir = "spectrograms"  # Directory to save spectrogram images
os.makedirs(save_dir, exist_ok=True)

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

# Read CSV
df = pd.read_csv(csv_path)

# Process each audio file
for index, row in df.iterrows():
    audio_path = row["Path"]
    genre = row["Genre"]
    
    # Load and convert to spectrogram
    aud = AudioUtil.open(audio_path)
    spec = AudioUtil.spectro_gram(aud)
    
    # Convert to numpy for visualization
    spec = spec.numpy()[0]  # Convert from tensor to numpy (mono channel)
    
    # Create save path
    genre_dir = os.path.join(save_dir, genre)  # Genre subdirectory
    os.makedirs(genre_dir, exist_ok=True)
    
    save_path = os.path.join(genre_dir, f"{index}.png")  # Image filename

    # Save spectrogram as an image
    plt.imsave(save_path, spec, cmap="inferno")

    print(f"Saved: {save_path}")

print("All spectrograms have been saved!")
