import librosa
import librosa.display
import numpy as np
import csv
import os

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    # Calculate MFCCs and pitches using a consistent hop length
    hop_length = 512
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hop_length)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=hop_length)
    
    # Prepare arrays to hold mean values per frame
    pitch_means = []
    for i in range(pitches.shape[1]):
        valid_pitches = pitches[:, i][magnitudes[:, i] > np.median(magnitudes[:, i])]
        pitch_mean = np.mean(valid_pitches) if valid_pitches.size > 0 else 0
        pitch_means.append(pitch_mean)
    
    mfccs_means = np.mean(mfccs, axis=1)

    return sr, pitch_means, mfccs_means.tolist()

def process_directory(directory_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['filename', 'frame'] + ['pitch_mean'] + ['mfcc_' + str(i) for i in range(13)]
        writer.writerow(headers)

        for filename in os.listdir(directory_path):
            if filename.endswith('.mp3'):
                file_path = os.path.join(directory_path, filename)
                sr, pitch_means, mfccs_means = extract_features(file_path)
                for i in range(len(pitch_means)):
                    writer.writerow([filename, i] + [pitch_means[i]] + mfccs_means)
                print("Processed:", filename)

# Example usage

process_directory('./audio-data', './csvdump/output.csv')