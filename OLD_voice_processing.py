import os
import json
import pyaudio
import wave
import librosa
from google.cloud import speech
from google.cloud import translate_v2 as translate

# Set your Google Cloud Credentials environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Admin\Downloads\stable-course-436011-p1-6f477ed37524.json"



# Function to record audio from microphone and save as WAV file
def record_audio(filename="audio.wav", duration=10):  # Increased recording duration to 10 seconds
    audio = pyaudio.PyAudio()
    
    # Open stream
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=16000, input=True, frames_per_buffer=1024)
    print("Recording...")

    frames = []
    for _ in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    print("Recording Finished.")

    # Stop stream and close
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save the recording to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        
    return filename

# Function to transcribe audio using Google Speech-to-Text API
def transcribe_audio(file_path):
    client = speech.SpeechClient()

    # Load audio file
    with open(file_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    # Configure to support multiple languages, prioritizing Tamil
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",  # Prioritize Tamil language
        alternative_language_codes=["en-US", "es", "fr", "hi", "zh","ta"]  # Add others as secondary options
    )

    try:
        response = client.recognize(config=config, audio=audio)

        # Extract transcript
        transcript = ""
        for result in response.results:
            transcript = result.alternatives[0].transcript
            print(f"Transcript: {transcript}")
            break  # We assume we're only interested in the first transcription

        return transcript

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# Function to detect language using Google Translate API and translate to English
def detect_and_translate(text, target_language="en"):
    translate_client = translate.Client()

    try:
        # Detect the language and translate
        result = translate_client.translate(text, target_language=target_language)
        detected_language = result['detectedSourceLanguage']
        translated_text = result['translatedText']
        
        print(f"Detected language: {detected_language}")
        print(f"Translated text: {translated_text}")

        return detected_language, translated_text

    except Exception as e:
        print(f"Error during translation: {e}")
        return None, None

# Function to extract basic audio features using librosa
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    
    # Extract pitch, tempo, energy, and duration as example features
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = sum(librosa.feature.rms(y=y)[0])
    
    return {
        "pitch": float(pitches.mean()),  # Convert NumPy float to Python float
        "duration": float(duration),
        "tempo": float(tempo),
        "energy": float(energy)
    }

# Main function to handle the entire flow and save result as JSON
def generate_json(file_path):
    # Step 1: Transcribe Audio
    transcript = transcribe_audio(file_path)
    
    if not transcript:
        print("Transcription failed.")
        result = {
            "error": "Failed to transcribe audio",
            "transcript": None,
            "translated_text": None,
            "characteristics": None
        }
    else:
        # Step 2: Detect Language and Translate to English
        detected_language, translated_text = detect_and_translate(transcript)

        # Step 3: Extract Audio Characteristics
        characteristics = extract_audio_features(file_path)

        # Step 4: Create JSON structure with original and translated text
        result = {
            "detected_language": detected_language if detected_language else "Unknown",  # If language detection fails
            "original_text": transcript,  # The transcribed text in the original language
            "translated_text": translated_text if translated_text else "Could not translate",  # English translation of the text
            "characteristics": characteristics  # Pitch, duration, tempo, energy, etc.
        }

    # Save to JSON file
    output_file = "output.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"JSON result saved to {output_file}")

if __name__ == "__main__":
    # Record an audio file (10 seconds by default now)
    audio_file = record_audio()

    # Generate JSON output
    generate_json(audio_file)
