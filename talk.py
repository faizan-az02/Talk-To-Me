import sounddevice as sd
import numpy as np
import torch
import requests

import subprocess
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

from piper import PiperVoice
from piper.config import SynthesisConfig

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

# -------------------------
# Load Piper
# -------------------------

print("Loading Piper...")

voice = PiperVoice.load("voices/en_US-hfc_female-medium.onnx")

print("Piper loaded.")

syn_config = SynthesisConfig(
    length_scale=0.75
)

# -------------------------
# Load Silero VAD
# -------------------------

print("Loading Silero VAD...")

vad_model = load_silero_vad()

print("Silero VAD loaded.")

# -------------------------
# Load Ollama
# -------------------------

print("Loading Ollama...")

session = requests.Session()

response = session.post("http://localhost:11434/api/chat",
                    json={
                        "model": "qwen3:8b",
                        "messages": [
                        {
                         "role": "system",
                        "content": "Reply with a \"Hi\""
                        }
                    ],
        "stream": False,
        "think": False,
        "keep_alive": -1,
    }       
)

print("Ollama loaded.")

text = ""
audio_chunks = []

CHUNK_SIZE = 512
SPEECH_THRESHOLD = 0.5
SILENCE_DURATION = 1
silence_samples = 0

speaking = False

subprocess.run("cls", shell=True)

print("Conversation Started...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=CHUNK_SIZE
) as stream:

    while text.strip().lower() not in ["exit.", "quit.", "bye."]:

        audio_chunks = []
        speaking = False

        while True:

            audio, overflowed = stream.read(CHUNK_SIZE)

            audio = np.squeeze(audio)

            audio_tensor = torch.from_numpy(audio)

            speech_probability = vad_model(
                audio_tensor,
                SAMPLE_RATE
            ).item()

            if speech_probability >= SPEECH_THRESHOLD:

                # We have speech again, so reset silence counter
                silence_samples = 0

                if not speaking:
                    speaking = True

            elif speaking:

                # We are currently in a speech session,
                # but this chunk is silence
                silence_samples += CHUNK_SIZE

                silence_seconds = silence_samples / SAMPLE_RATE

                if silence_seconds >= SILENCE_DURATION:
                    break

            if speaking:
                audio_chunks.append(audio.copy())                        

        complete_audio = np.concatenate(audio_chunks)

        segments, info = model.transcribe(
            complete_audio,
            language="en"
        )

        text = ""

        for segment in segments:
            text += segment.text

        print("You:", text.strip())

        response = session.post("http://localhost:11434/api/chat",
                                json={
                                    "model": "qwen3:8b",
                                    "messages": [
                                        {
                                            "role": "system",
                                            "content": "You are a helpful assistant. Answer in at MAX 3 lines."
                                        },
                                        {
                                            "role": "user",
                                            "content": text.strip()
                                        }
                                    ],
                                "stream": False,
                                "think": False,
                                "keep_alive": -1,
            }       
        )

        response_text = response.json()["message"]["content"]

        print("System:", response_text)

        for chunk in voice.synthesize(response_text, syn_config=syn_config):

            sd.play(
                chunk.audio_float_array,
                samplerate=chunk.sample_rate
            )

            sd.wait()