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
def audio_callback(indata, frames, time, status):
    if status:
        logging.warning("Stream status: %s", status)
    try:
        sf_writer_input.write(indata[:, 0])
        # Directly write input to the file to test speed issues
        features = extract_features(indata[:, 0])
        adjusted_features = model.predict(np.expand_dims(features, axis=0))
        adjusted_audio = librosa.effects.pitch_shift(indata[:, 0], sr=22050, n_steps=adjusted_features[0])
        outdata[:] = adjusted_audio.reshape(-1, 1)
        sf_writer.write(outdata[:, 0])  # Write the raw audio data to the file
        logging.info("Audio data written to file.")
    except Exception as e:
        logging.error("Error processing audio frame: %s", str(e))

def run_stream():
    try:
        # Set up the audio stream
        with sd.InputStream(callback=audio_callback, dtype='float32', channels=1, samplerate=44100):
            logging.info("Starting audio stream...")
            sd.sleep(sys.maxsize)  # Keep the stream open
    except Exception as e:
        logging.error("Stream error: %s", str(e))
    finally:
        logging.info("Stopping audio stream.")
        sf_writer.close()

if __name__ == "__main__":
    run_stream()
