# Talk-To-Me

Talk-To-Me is a local voice assistant prototype. It listens through your microphone, detects when you are speaking, transcribes your speech with Whisper, sends the text to a local Ollama model, and speaks the response back with Piper text-to-speech.

The project is designed to run fully on your machine.

## Features

- Microphone input with `sounddevice`
- Voice activity detection with Silero VAD
- Speech-to-text with `faster-whisper`
- Local chat responses through Ollama
- Text-to-speech with Piper
- Bundled Piper voice models in `voices/`
- Simple exit flow using spoken commands like `exit.`, `quit.`, or `bye.`

## Project Structure

```text
Talk-To-Me/
+-- talk.py                  # Main local voice conversation loop
+-- voices/                  # Piper ONNX voice models and metadata
+-- README.md
```

## Requirements

- Python 3.10 or newer
- A working microphone and speaker
- Ollama running locally
- The Ollama model used by `talk.py`: `qwen3:8b`

## Setup

Install the Python dependencies:

```bash
pip install sounddevice numpy torch requests faster-whisper silero-vad piper-tts
```

Install Ollama, then pull the model:

```bash
ollama pull qwen3:8b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

## Usage

Run the assistant:

```bash
python talk.py
```

When the script starts, it loads:

1. Whisper for transcription
2. Piper for speech synthesis
3. Silero VAD for speech detection
4. Ollama for local responses

After `Conversation Started...` appears, speak into your microphone. The assistant waits until you stop talking, transcribes your speech, gets a response from Ollama, and plays the response aloud.

To stop the conversation, say one of:

```text
exit.
quit.
bye.
```

## Voice Models

The current script uses:

```python
voices/en_US-hfc_female-medium.onnx
```

Other bundled voices include:

- `en_US-amy-medium`
- `en_US-kristin-medium`
- `en_US-lessac-medium`

To switch voices, update the `PiperVoice.load(...)` path in `talk.py`.

## Configuration

Useful settings are near the top of `talk.py`:

```python
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
```

The active Ollama model is configured in the chat request body:

```python
"model": "qwen3:8b"
```

The current assistant instruction asks the model to answer in at most three lines.

## Troubleshooting

If Ollama requests fail, make sure Ollama is running at:

```text
http://localhost:11434
```

If audio input or playback fails, check that your default microphone and speaker are available to Python. On some systems, `sounddevice` may require PortAudio support.

If transcription is slow, try a smaller Whisper model or use a GPU-supported configuration.

If Piper cannot load a voice, make sure the `.onnx` file and its matching `.onnx.json` metadata file are both present in `voices/`.

## Roadmap Notes

`plan.md` sketches a possible future architecture with WebRTC, FastAPI, Pipecat, browser audio, and the same local STT/LLM/TTS stack.
