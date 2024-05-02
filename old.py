import numpy as np
import sounddevice as sd
import librosa
from keras.models import load_model
import logging
import queue
import sys

# Setup logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
try:
    model = load_model('accent_modification_model.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Failed to load model: %s", str(e))
    sys.exit(1)

# Define a queue to introduce delay
audio_buffer = queue.Queue(maxsize=10)

def audio_callback(indata, outdata, frames, time, status):
    if status:
        logging.warning("Stream status: %s", status)
    try:
        # Assuming the input is mono for simplicity
        features = extract_features(indata[:, 0])
        features = np.expand_dims(features, axis=0)  # Ensure batch dimension
        if features.shape[1] != 14:
            raise ValueError(f"Expected feature shape (1, 14), but got {features.shape}")
        
        # Make a prediction
        adjusted_features = model.predict(features)
        
        # Add a delay to output
        if audio_buffer.full():
            outdata[:] = audio_buffer.get_nowait()
        else:
            outdata.fill(0)
        
        # Modify the input audio before putting it in the buffer
        modified_audio = indata * 0.01  # Reduce volume to prevent feedback
        audio_buffer.put_nowait(modified_audio)

        logging.info("Processed audio with adjusted features: %s", adjusted_features)
    except Exception as e:
        logging.error("Error processing audio frame: %s", str(e))
        outdata.fill(0)  # Zero out the output buffer if there's an error

def extract_features(audio_data):
    # Extract MFCC or any other feature used by your model
    mfcc = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13)
    # Append additional features to match the expected input shape of (1, 14)
    zero_crossings = np.sum(librosa.zero_crossings(audio_data))
    feature_vector = np.append(np.mean(mfcc, axis=1), zero_crossings)
    return feature_vector

def run_stream():
    try:
        # Set up the audio stream
        with sd.Stream(callback=audio_callback, dtype='float32', channels=1,samplerate=22050):
            logging.info("Starting audio stream...")
            sd.sleep(sys.maxsize)  # Keep the stream open
    except Exception as e:
        logging.error("Stream error: %s", str(e))
    finally:
        logging.info("Stopping audio stream.")

if __name__ == "__main__":
    run_stream()
