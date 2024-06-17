import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation as aS
from pydub import AudioSegment
from pydub.utils import make_chunks
from multiprocessing import Pool, cpu_count

def downsample_audio(audio_segment, target_rate):
    return audio_segment.set_frame_rate(target_rate)

# Process Chunks
def process_chunk(chunk_info):
    chunk, i, target_rate = chunk_info
    chunk_filename = f"chunk_{i}.wav"
    chunk.export(chunk_filename, format="wav");

    # Read the chunk audio file
    [Fs, x] = audioBasicIO.read_audio_file(chunk_filename)
    x = audioBasicIO.stereo_to_mono(x)

    # Parameters for silence removal
    st_win = 0.05  # Window size (in seconds)
    st_step = 0.05  # Step size (in seconds)
    smooth_window = 1.0  # Smoothing window (in seconds)
    weight = 0.3  # Weight factor for silence removal

    # Perform silence removal
    segments = aS.silence_removal(x, Fs, st_win, st_step, smooth_window, weight)

    # Save segmented audio files
    for j, segment in enumerate(segments):
        start_time = segment[0]
        end_time = segment[1]
        extracted_segment = chunk[start_time * 1000:end_time * 1000]  # Convert to milliseconds
        extracted_segment.export(f"chunk_{i}_segment_{j}.wav", format="wav")

    return f"Chunk {i} processed and segments saved."

# Main script
if __name__ == "__main__":
    # Parameters
    filename = "audio_book.wav"
    chunk_length_ms = 3600000  # 1-hour chunks
    target_rate = 16000  # Target sampling rate for downsampling

    # Load audiobook audio
    audio_segment = AudioSegment.from_wav(filename)

    # Downsample the audio
    audio_segment = downsample_audio(audio_segment, target_rate)

    # Split the audio into 1-hour chunks
    chunks = make_chunks(audio_segment, chunk_length_ms)

    # Prepare the chunk info for parallel processing
    chunk_info_list = [(chunk, i, target_rate) for i, chunk in enumerate(chunks)]

    # Use multiprocessing Pool to process chunks in parallel
    pool = Pool(cpu_count())
    results = pool.map(process_chunk, chunk_info_list)

    for result in results:
        print(result)

    print("All chunks processed and segments saved.")



