import sys
import logging
import numpy as np
import sounddevice as sd
import librosa
from keras.models import load_model

# Setup logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
try:
    model = load_model('accent_modification_model.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Failed to load model: %s", str(e))
    sys.exit(1)

# Define the audio stream callback function
def audio_callback(indata, outdata, frames, time, status):
    if status:
        logging.warning("Stream status: %s", status)
    # Convert audio input to appropriate feature vector
    try:
        # Assuming the input is mono for simplicity
        features = extract_features(indata[:, 0])
        features = np.expand_dims(features, axis=0)  # Model expects a batch dimension
        prediction = model.predict(features)
        # Here you might modify 'indata' based on 'prediction' before outputting
        modified_audio = modify_audio(indata, prediction)  # Implement this function based on your model's output
        outdata[:] = modified_audio  # Output the modified audio
        stream = sd.Stream(callback=audio_callback)
        stream.start()
        logging.info("Processed audio with prediction: %s", prediction)
    except Exception as e:
        logging.error("Error processing audio frame: %s", str(e))

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
        with sd.InputStream(callback=audio_callback):
            logging.info("Starting audio stream...")
            sd.sleep(sys.maxsize)  # Keep the stream open
    except Exception as e:
        logging.error("Stream error: %s", str(e))
    finally:
        logging.info("Stopping audio stream.")

if __name__ == "__main__":
    run_stream()