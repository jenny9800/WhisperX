import whisperx

# Load the Whisper model
model = whisperx.load_model("base", device="cpu", compute_type="int8")

# Path to the audio file
audio_path = "/Users/jennyulee/Downloads/audioEx.wav"

# Transcribe the audio file
result = model.transcribe(audio_path)

# Print the transcription
print("Transcription Result:\n", result)
