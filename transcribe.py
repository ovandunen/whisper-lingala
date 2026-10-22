#!/usr/bin/env python3
"""Transcribe audio files using the fine-tuned Whisper model baked into the image."""
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_DIR = "/app/model"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model from {MODEL_DIR} (device={device})...", file=sys.stderr, flush=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
print("Model ready.", file=sys.stderr, flush=True)

if len(sys.argv) < 2:
    print("Usage: docker run --rm -v /path/to/audio:/audio whisper-runner /audio/file.mp3 [more files...]")
    sys.exit(1)

for audio_path in sys.argv[1:]:
    print(f"\n--- {audio_path} ---", file=sys.stderr, flush=True)
    result = pipe(audio_path)
    print(f"{audio_path}\t{result['text']}")
