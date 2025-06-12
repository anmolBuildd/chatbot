import whisper
import os

INPUT_FILE = "seminar.mp3" 
OUTPUT_FILE = "data/seminar_transcript.txt"

# Create output folder
os.makedirs("data", exist_ok=True)

print("🔁 Loading Whisper model...")
model = whisper.load_model("large")  

print(f"🎧 Transcribing {INPUT_FILE}...")
result = model.transcribe(INPUT_FILE)

# Save the transcript
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"✅ Transcription complete: {OUTPUT_FILE}")
