import torchaudio
import os

#programul imparte inregistrarea in segmente de durata specificata 4 secunde
input_dir = "inregistrari"
output_dir = "inregistraritrim"

def split_audio(audio_file, output_dir, original_name=""):
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
    except RuntimeError:
        print(f"Failed to load: {audio_file}")
        return

    total_duration = waveform.size(1) / sample_rate
    segment_duration = 4  # Split at 4 seconds
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_segments = int(total_duration / segment_duration)

    for i in range(num_segments):
        start = int(i * segment_duration * sample_rate)
        end = int((i + 1) * segment_duration * sample_rate)
        if end > waveform.size(1):
            end = waveform.size(1)
        segment_waveform = waveform[:, start:end]
        segment_name = f"{original_name}_segment_{i + 1}.wav"
        output_file = os.path.join(output_dir, segment_name)
        torchaudio.save(output_file, segment_waveform, sample_rate)

def split_audio_files(input_dir, output_dir):
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_dir, class_folder)
            os.makedirs(output_class_path, exist_ok=True)
            for audio_file in os.listdir(class_path):
                audio_file_path = os.path.join(class_path, audio_file)
                if os.path.isfile(audio_file_path):
                    split_audio(audio_file_path, output_class_path, audio_file)

split_audio_files(input_dir, output_dir)