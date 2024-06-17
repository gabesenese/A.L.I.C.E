import os
from pocketsphinx import AudioFile
import nltk
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Load CMU Pronouncing Dictionary
nltk.download('cmudict')
d = nltk.corpus.cmudict.dict()

# Directory containing segmented audio files & checkpoint
segments_dir = "./segments/"
phoneme_output_dir = "./phonemes/"

# Ensure phoneme output directory exists
if not os.path.exists(phoneme_output_dir):
    os.makedirs(phoneme_output_dir)

# Config decoder's paths
model_dir = "C:/Users/gabri/AppData/Local/Programs/Python/Python312/Lib/site-packages/pocketsphinx/model/en-us"
hmm_path = "C:/Users/gabri/AppData/Local/Programs/Python/Python312/Lib/site-packages/pocketsphinx/model/en-us/en-us"
dict_path = "C:/Users/gabri/AppData/Local/Programs/Python/Python312/Lib/site-packages/pocketsphinx/model/en-us/cmudict-en-us.dict"
lm_path = "C:/Users/gabri/AppData/Local/Programs/Python/Python312/Lib/site-packages/pocketsphinx/model/en-us/en-us.lm.bin"

# Configure the decoder
config = {
    'verbose': True,
    'audio_file': None,
    'buffer_size': 2048,
    'no_search': False,
    'full_utt': False,
    'hmm': hmm_path,
    'lm': lm_path,
    'dict': dict_path,
    'bestpath': True
}

def words_to_phonemes(words):
    phoneme_list = []
    for word in words:
        if word.lower() in d:
            phonemes = d[word.lower()][0]  # Take the first pronunciation variant
            phoneme_list.extend(phonemes)
        else:
            print(f"Word '{word}' not found in CMU dictionary.")
    return phoneme_list

def process_file(filename):
    if filename.endswith(".wav"):
        audio_file_path = os.path.join(segments_dir, filename)
        config['audio_file'] = audio_file_path

        try:
            # Create an AudioFile object with the current configuration
            audio = AudioFile(**config)

            # Process the audio file to extract phonemes
            for phrase in audio:
                if phrase.hyp():
                    words = phrase.hypothesis().split()
                    phonemes = words_to_phonemes(words)
                    segment_audio = AudioSegment.from_wav(audio_file_path)
                    
                    # Save each phoneme audio segment
                    for i, phoneme in enumerate(phonemes):
                        phoneme_file = os.path.join(phoneme_output_dir, f"{phoneme}_{i}.wav")
                        segment_audio.export(phoneme_file, format="wav")

            # Print progress bar
            pbar.update(1)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

def main():
    print("Starting to generate phonemes ...")
    files = [f for f in os.listdir(segments_dir) if f.endswith(".wav")]
    
    # Initialize progress bar
    with tqdm(total=len(files)) as pbar:
        # Use multiprocessing to process files concurrently
        with Pool(cpu_count()) as pool:
            pool.map(process_file, files)
    
    print("Phoneme extraction completed.")

if __name__ == "__main__":
    main()
