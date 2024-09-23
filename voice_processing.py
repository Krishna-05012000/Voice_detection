


import os
import io
import json
import librosa
import numpy as np
from google.cloud import speech, translate_v2 as translate, language_v1 as language
import pyaudio
import wave


# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Admin\Downloads\stable-course-436011-p1-a7b0befef938.json'

# Initialize Google Cloud Clients
speech_client = speech.SpeechClient()
translate_client = translate.Client()
language_client = language.LanguageServiceClient()

# Function to record audio
def record_audio(filename, duration=10, rate=16000, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=1024)
    print(f"Recording for {duration} seconds...")
    frames = []
    
    for _ in range(int(rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    print("Recording completed.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded data as a .wav file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to transcribe audio and detect language
def transcribe_audio(file_path):
    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",  # Default language (adjust as needed)
        alternative_language_codes=['ta-IN', 'hi-IN', 'es', 'fr', 'de']  # Tamil, Hindi, Spanish, etc.
    )
    
    response = speech_client.recognize(config=config, audio=audio)
    
    if not response.results:
        print("No speech detected or language is unclear.")
        return ""
    
    transcript = response.results[0].alternatives[0].transcript
    print(f"Transcript: {transcript}")
    return transcript

# Function to detect the language of the text
def detect_language(text):
    if not text:
        return "und"  # Return 'und' for undefined if no text was transcribed
    result = translate_client.detect_language(text)
    detected_lang = result["language"]
    print(f"Detected language: {detected_lang}")
    return detected_lang

# Function to translate text to English
def translate_text(text, target_language="en"):
    if not text:
        return ""
    translation = translate_client.translate(text, target_language=target_language)
    translated_text = translation["translatedText"]
    print(f"Translated text: {translated_text}")
    return translated_text

# Function to analyze text sentiment (optional, based on your requirement)
def analyze_sentiment(text):
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
    sentiment = language_client.analyze_sentiment(document=document).document_sentiment
    print(f"Sentiment: Score: {sentiment.score}, Magnitude: {sentiment.magnitude}")
    return sentiment

# Function to extract audio characteristics (pitch, loudness)
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Estimate pitch using librosa's pyin method
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nanmean(f0)  # Mean pitch across the file
    
    rms = librosa.feature.rms(y=y)
    loudness = np.mean(rms)
    
    print(f"Audio Characteristics: Duration: {duration}, Pitch: {pitch}, Loudness: {loudness}")
    return duration, pitch, loudness

# Generate JSON output
def generate_json(file_path):
    transcript = transcribe_audio(file_path)
    detected_language = detect_language(transcript)
    translated_text = translate_text(transcript)
    
    duration, pitch, loudness = extract_audio_features(file_path)
    
    # Convert numpy.float32 to Python float
    output = {
        "transcript": transcript,
        "detected_language": detected_language,
        "translated_text": translated_text,
        "audio_features": {
            "duration": float(duration),  # Ensure it's a standard Python float
            "pitch": float(pitch) if pitch is not None else None,  # Some pitch values might be NaN
            "loudness": float(loudness)
        }
    }
    
    # Save the output to a JSON file
    with open("output_result.json", "w") as json_file:
        json.dump(output, json_file, indent=4)
    print("JSON file generated.")


if __name__ == "__main__":
    audio_file = "audio_sample.wav"  # Recorded or existing audio file
    record_audio(audio_file, duration=15)  # Record audio for 15 seconds
    generate_json(audio_file)  # Process the audio and generate the output JSON
