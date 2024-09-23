import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import json
import librosa

# Load the Whisper model - using the large model for better accuracy in multilingual tasks
model = whisper.load_model("large")

def record_audio(duration=15, filename="output.wav"):
    """
    Record audio for the specified duration (in seconds) and save to a file.
    """
    print(f"Recording for {duration} seconds...")
    fs = 16000  # Sample rate
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # Wait until recording is finished
    wav.write(filename, fs, np.int16(audio * 32767))  # Save as WAV file
    print("Recording completed.")
    return filename  # Return the file path for further processing

def transcribe_audio(file_path):
    """
    Transcribe the audio and detect the language using Whisper.
    """
    # Transcribe the audio with automatic language detection
    print("Transcribing audio...")
    result = model.transcribe(file_path, task="transcribe", language=None)  # Automatic language detection
    
    transcript = result['text']
    detected_language = result['language']  # Detected language in ISO 639-1 code (e.g., "ta" for Tamil)

    print(f"Transcript: {transcript}")
    print(f"Detected language: {detected_language}")
    
    return transcript, detected_language

def translate_text(file_path, target_language="en"):
    """
    Translate the given audio into the target language (default is English).
    """
    result = model.transcribe(file_path, task="translate", language=None)  # Translate to English
    translated_text = result['text']
    
    print(f"Translated text: {translated_text}")
    return translated_text

def extract_audio_features(file_path):
    """
    Extract basic audio features like duration, pitch, and loudness.
    """
    print(f"Extracting audio features from {file_path}...")
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Extract pitch using librosa.pyin()
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )

    # Compute the median pitch of the voiced frames
    if f0 is not None and np.any(np.isfinite(f0)):
        pitch = np.nanmedian(f0)
    else:
        pitch = None

    # Compute loudness using RMS
    rms = librosa.feature.rms(y=y)
    loudness = float(np.mean(rms))

    print(f"Duration: {duration} seconds")
    print(f"Pitch: {pitch}")
    print(f"Loudness: {loudness}")
    
    return duration, pitch, loudness

def generate_json(file_path):
    """
    Generate a JSON output with the transcription, detected language, translation, and audio features.
    """
    transcript, detected_language = transcribe_audio(file_path)
    translated_text = translate_text(file_path)
    duration, pitch, loudness = extract_audio_features(file_path)
    
    output = {
        "transcript": transcript,
        "detected_language": detected_language,
        "translated_text": translated_text,
        "audio_features": {
            "duration": duration,
            "pitch": pitch,
            "loudness": loudness
        }
    }

    # Convert any NumPy types to native Python types for JSON serialization
    output = json.loads(json.dumps(output, default=lambda x: x.item() if isinstance(x, np.generic) else x))
    
    # Save the output in a JSON file
    json_file_path = "output.json"
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=4, ensure_ascii=False)
    
    print(f"Output saved to {json_file_path}")

if __name__ == "__main__":

    audio_file = record_audio(duration=15, filename="recorded_audio.wav")

    generate_json(audio_file)
