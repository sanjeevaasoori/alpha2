import numpy as np
import sounddevice as sd
import librosa
from keras.models import load_model
import logging
import sys
import soundfile as sf

# Setup logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
try:
    model = load_model('accent_modification_model.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Failed to load model: %s", str(e))
    sys.exit(1)

# Define the output file
output_file = 'modified_audio_output.wav'
output_file_input = 'input_audio.wav'
sf_writer = sf.SoundFile(output_file, mode='w', samplerate=44100, channels=1)
sf_writer_input = sf.SoundFile(output_file_input, mode='w', samplerate=44100, channels=1)

def extract_features(audio_data, sample_rate=44100):
    # Extract MFCCs or any other feature used by your model
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    # Calculate additional features, e.g., zero crossings
    zero_crossings = np.sum(librosa.zero_crossings(audio_data))
    # Combine all features into a single vector
    feature_vector = np.append(np.mean(mfcc, axis=1), zero_crossings)
    return feature_vector

def audio_callback(indata, outdata, frames, time, status):
    if status:
        logging.warning("Stream status: %s", status)
    try:
        sf_writer_input.write(indata[:, 0])
        features = extract_features(indata[:, 0])
        adjusted_features = model.predict(np.expand_dims(features, axis=0))
        adjusted_audio = librosa.effects.pitch_shift(indata[:, 0], sr=44100, n_steps=adjusted_features[0])
        outdata[:] = adjusted_audio.reshape(-1, 1) 
        sf_writer.write(outdata[:, 0]) 
    except Exception as e:  
        logging.error("Error in audio callback: %s", str(e)) 
        outdata.fill(0)  # Optional: Fill with zeros if an error occurs

def run_stream():
    with sd.InputStream(callback=audio_callback, dtype='float32', channels=1, samplerate=44100) as stream:
        logging.info("Starting audio stream...")
        stream.start()
        try:
            while stream.active:  
                sd.sleep(1000)  # Sleep for a second
        except KeyboardInterrupt:
            print('\nInterrupted by user')
        finally:
            stream.stop()  
            logging.info("Stopping audio stream.")
            sf_writer.close()
            sf_writer_input.close()

if __name__ == "__main__":
    run_stream()
