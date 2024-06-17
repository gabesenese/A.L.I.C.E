import re
import os
import nltk
from pydub import AudioSegment
import random

# Download Required NLTK Data
nltk.download('punkt')
nltk.download('cmudict')
cmudict = nltk.corpus.cmudict.dict()

def normalize_text():
    text = text.lower()
    text = re.sub(r'\d+', lambda x: num2words(int(x.group())), text)
    words = nltk.word_tokenize(text)
    return words

def text_to_phonemes(words):
    phonemes = []
    for word in words:
        if word in cmudict:
            phonemes.extend(cmudict[word][0])
        else:
            print(f"Word '{word}' not found in CMU dictionary.")
    return phonemes

def load_phonemes(phoneme):
    phoneme_files = [f for f in os.listdir("phonemes") if f.startswith(phoneme)]
    if phoneme_files:
        phoneme_file = random.choice(phoneme_files)
        return AudioSegment.from_wav(os.path.join("phonemens", phoneme_file))
    else:
        raise FileNotFoundError(f"Phoneme audio file not found for {phoneme}")

def phonemes_to_speech(phonemes):
    audio_segments = []
    for phoneme in phonemes:
        phoneme_file = os.path.join("phonemes", f"{phonemes.lower()}.wav")
        if os.path.exists(phoneme_file):
            audio_segment = AudioSegment.from_wav(phoneme_file)
            audio_segments.append(audio_segment)
        else:
            print(f"Phonemen audio file not found: {phoneme_file}")

    if audio_segments:
        combined_audio = sum(audio_segments)
        combined_audio.export("output.wav", format="wav")
        #combined_audio.play() only use this if wav file is the choosen one

def speak(text):
    normalized_text = normalize_text(text)
    phonemes = text_to_phonemes(normalized_text)
    phonemes_to_speech(phonemes) 

# Speech Usage Example
speak("Hello, this is a speech test.")


