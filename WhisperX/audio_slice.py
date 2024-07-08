import whisperx
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

# Load the Whisper model with appropriate data type
model = whisperx.load_model("base", device="cpu", compute_type="int8")

# Path to the audio file
audio_path = "/Users/jennyulee/Downloads/audioEx.wav"

# Function to slice audio file into chunks
def slice_audio(audio_path, chunk_length_ms=30000):  # 30 seconds
    audio = AudioSegment.from_file(audio_path)
    chunks = make_chunks(audio, chunk_length_ms)
    chunk_paths = []

    # Create a temporary directory to save chunks
    temp_dir = "/Users/jennyulee/Desktop/WhisperX/temp_chunks"
    os.makedirs(temp_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(temp_dir, f"chunk{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    
    return chunk_paths, temp_dir

# Slice the audio file into chunks
chunk_paths, temp_dir = slice_audio(audio_path)

# Transcribe each chunk and concatenate the results
full_transcription = ""

for chunk_path in chunk_paths:
    result = model.transcribe(chunk_path)
    
    # Print the entire result to understand its structure
    print(f"Result for {chunk_path}:\n", result)
    
    # Access and print the transcription text
    transcription = result.get("text", "Transcription not found.")
    full_transcription += transcription + " "

# Print the full transcription
print("Full Transcription:\n", full_transcription)

# Optionally, remove temporary chunk files
for chunk_path in chunk_paths:
    os.remove(chunk_path)
os.rmdir(temp_dir)
