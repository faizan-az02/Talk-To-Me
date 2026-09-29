import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# -------------------------
# Settings
# -------------------------

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

# -------------------------
# Load Whisper
# -------------------------

print("Loading Whisper...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded.")

text = ""

# -------------------------
# Record microphone
# -------------------------
while text.strip().lower() != "exit.":

    print(f"\nSpeak now... recording for {RECORD_SECONDS} seconds.")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = np.squeeze(audio)

    print("Recording finished.")
    print("Transcribing...\n")

    # -------------------------
    # Transcribe
    # -------------------------

    segments, info = model.transcribe(
        audio,
        language="en"
    )

    text = ""

    for segment in segments:
        text += segment.text

    print("You said:")
    print(text.strip())