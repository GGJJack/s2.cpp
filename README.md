# s2.cpp

> **ALPHA — EXPERIMENTAL SOFTWARE**
> This is an early-stage, community-built inference engine. Expect rough edges, missing features, and breaking changes. Not production-ready.

**s2.cpp** — Fish Audio's S2 Pro Dual-AR text-to-speech model running locally via a pure C++/GGML inference engine with CPU, Vulkan, and CUDA GPU backends. No Python runtime required after build.

> **Built on Fish Audio S2 Pro**
> The model weights are licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.
> See [LICENSE.md](LICENSE.md) for full terms. Commercial use requires a separate license from Fish Audio — contact [business@fish.audio](mailto:business@fish.audio).

---

## What this is

This repository contains:

- **`s2.cpp`** — a self-contained C++17 inference engine built on [ggml](https://github.com/ggml-org/ggml), handling tokenization, Dual-AR generation, audio codec encode/decode, and WAV output with no Python dependency
- **`tokenizer.json`** — Qwen3 BPE tokenizer with ByteLevel pre-tokenization
- GGUF model files are **not included** here — see [Model variants](#model-variants) below

The engine runs the full pipeline: text → tokens → Slow-AR transformer (with KV cache) → Fast-AR codebook decoder → audio codec → WAV file.

---

## Model variants

GGUF files are available at [rodrigomt/s2-pro-gguf](https://huggingface.co/rodrigomt/s2-pro-gguf) on Hugging Face.

| File | Size | Notes |
|---|---|---|
| `s2-pro-f16.gguf` | 9.3 GB | Full precision — reference quality |
| `s2-pro-q8_0.gguf` | 5.3 GB | Near-lossless — recommended for 8+ GB VRAM |
| `s2-pro-q6_K.gguf` | 4.3 GB | Good quality/size balance — recommended for 6+ GB VRAM |
| `s2-pro-q5_K_M.gguf` | 3.8 GB | Smaller with still-good quality |
| `s2-pro-q4_K_M.gguf` | 3.4 GB | Best compact variant so far in quick RU validation |
| `s2-pro-q3_K.gguf` | 2.9 GB | Usable, but starts stretching short words |
| `s2-pro-q2_K.gguf` | 2.4 GB | Lowest-size experimental variant |

All variants include both the transformer weights and the audio codec in a single file.
The quantized variants above were regenerated with the codec tensors (`c.*`) kept in `F16`, so only the AR transformer is quantized.

---

## Requirements

### Build dependencies

- CMake ≥ 3.14
- C++17 compiler (GCC ≥ 10, Clang ≥ 11, MSVC 2019+)
- For Vulkan GPU support: Vulkan SDK and `glslc`
- For CUDA/NVIDIA GPU support: CUDA Toolkit ≥ 12.4
  - **MSVC 2019+ note:** MSVC 2019 and later require CUDA ≥ 12.4 when building GGML. Older CUDA versions will produce compiler compatibility errors; upgrade to 12.4+ to resolve them.

```bash
# Ubuntu / Debian
sudo apt install cmake build-essential

# Vulkan (optional, for AMD/Intel GPU acceleration)
sudo apt install vulkan-tools libvulkan-dev glslc

# CUDA (optional, for NVIDIA GPU acceleration)
# Install from https://developer.nvidia.com/cuda-downloads
```

### Runtime

No Python or PyTorch required. The binary links only against the ggml shared libraries built alongside it.

---

## Building

Clone with submodules (ggml is a submodule):

```bash
git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git
cd s2.cpp
```

### CPU only

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

### With Vulkan GPU support (AMD/Intel)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON
cmake --build build --parallel $(nproc)
```

### With CUDA GPU support (NVIDIA)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_CUDA=ON
cmake --build build --parallel $(nproc)
```

The binary is produced at `build/s2`.

---

## Usage

### Basic synthesis (CPU)

```bash
./build/s2 \
  -m s2-pro-q6_K.gguf \
  -t tokenizer.json \
  -text "The quick brown fox jumps over the lazy dog." \
  -o output.wav
```

`tokenizer.json` is searched automatically in the same directory as the model file, then the parent directory, then the working directory.

### Voice cloning with a reference audio

Provide a short reference clip (5–30 seconds, WAV or MP3) and a transcript of it:

```bash
./build/s2 \
  -m s2-pro-q6_K.gguf \
  -t tokenizer.json \
  -pa reference.wav \
  -pt "Transcript of what the reference speaker says." \
  -text "Now synthesize this text in that voice." \
  -o output.wav
```

By default, the engine keeps a small `8`-token floor before `EOS`, trims trailing silence from the final WAV, and peak-normalizes the output to `0.95`. All three behaviors are optional and can be overridden from the CLI.

### GPU inference via Vulkan (AMD/Intel)

```bash
./build/s2 \
  -m s2-pro-q6_K.gguf \
  -t tokenizer.json \
  -text "Text to synthesize." \
  -v 0 \
  -o output.wav
```

`-v 0` selects the first Vulkan device.

### GPU inference via CUDA (NVIDIA)

```bash
./build/s2 \
  -m s2-pro-q6_K.gguf \
  -t tokenizer.json \
  -text "Text to synthesize." \
  -c 0 \
  -o output.wav
```

`-c 0` selects the first CUDA device. The transformer runs on GPU; the audio codec always runs on CPU (executes only twice per synthesis).

### Server

```bash
./build/s2 \
  -m s2-pro-q6_k.gguf \
  -t tokenizer.json \
  --vulkan 0 \
  --server 1 \
  -H "127.0.0.1" \
  -P 3000

curl --location 'http://127.0.0.1:3000/generate' \
  --form 'reference=928496226.wav' \
  --form 'reference_text="Well... I was joking, but your conspicuous silence has been noted."' \
  --form 'text="Hello World?"' \
  --form 'params="{\"max_new_tokens\":2048,\"temperature\":0.58,\"top_p\":0.88,\"top_k\":40}"'
```

### All options

| Flag | Default | Description |
|---|---|---|
| `-m`, `--model` | `model.gguf` | Path to GGUF model file |
| `-t`, `--tokenizer` | `tokenizer.json` | Path to tokenizer.json |
| `-text` | `"Hello world"` | Text to synthesize |
| `-pa`, `--prompt-audio` | — | Reference audio file for voice cloning (WAV/MP3) |
| `-pt`, `--prompt-text` | — | Transcript of the reference audio |
| `-o`, `--output` | `out.wav` | Output WAV file path |
| `-v`, `--vulkan` | `-1` (CPU) | Vulkan device index (`-1` = CPU only) |
| `-c`, `--cuda` | `-1` (CPU) | CUDA device index (`-1` = CPU only) |
| `-threads N` | `4` | Number of CPU threads |
| `-max-tokens N` | `512` | Max tokens to generate (~21s of audio per 440 tokens) |
| `--min-tokens-before-end N` | `8` | Minimum generated tokens before `EOS` is allowed; use `0` to allow immediate stop |
| `-temp F` | `0.7` | Sampling temperature |
| `-top-p F` | `0.7` | Top-p nucleus sampling |
| `-top-k N` | `30` | Top-k sampling |
| `--trim-silence` / `--no-trim-silence` | `trim` enabled | Enable or disable trailing silence trimming on the saved WAV |
| `--normalize` / `--no-normalize` | `normalize` enabled | Enable or disable peak normalization to `0.95` on the saved WAV |
| `--server` | — | Start HTTP server instead of CLI synthesis |
| `-H`, `--host` | `127.0.0.1` | Server bind address |
| `-P`, `--port` | `3030` | Server port |

Lower `--min-tokens-before-end` values reduce forced tail padding but increase the chance of very short outputs. Setting it to `0` gives the sampler full freedom to end immediately.

---

### HTTP server mode

Start the server:

```bash
./build/s2 -m s2-pro-q6_K.gguf --server
# or with custom host/port:
./build/s2 -m s2-pro-q6_K.gguf --server -H 0.0.0.0 -P 8080
```

**`POST /generate`** — synthesize audio (multipart/form-data)

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Text to synthesize |
| `reference` | file | no | Reference WAV for voice cloning |
| `reference_text` | string | no | Transcript of the reference audio |
| `params` | JSON string | no | Generation params: `max_new_tokens`, `temperature`, `top_p`, `top_k` |

Returns `audio/wav`.

```bash
# Basic
curl -X POST http://127.0.0.1:3030/generate \
  --form "text=Hello world" \
  --form 'params={"max_new_tokens":512,"temperature":0.58,"top_p":0.88,"top_k":40}' \
  -o output.wav

# With voice cloning
curl -X POST http://127.0.0.1:3030/generate \
  --form "reference=@reference.wav" \
  --form "reference_text=Transcript of the reference." \
  --form "text=Text to synthesize in that voice." \
  --form 'params={"max_new_tokens":512,"temperature":0.58,"top_p":0.88,"top_k":40}' \
  -o output.wav
```

---

## Choosing a model

| VRAM available | Recommended model |
|---|---|
| ≥ 10 GB | `q8_0` — near-lossless quality |
| 6–9 GB | `q6_K` — good quality/size balance |
| 5–7 GB | `q4_K_M` — best compact variant in current quick validation |
| < 5 GB | `q3_K` or `q2_K` — experimental, quality drops faster |

VRAM usage at runtime is approximately equal to the file size (transformer weights only; codec runs on CPU).

---

## Architecture notes

S2 Pro uses a **Dual-AR** architecture:

- **Slow-AR** — a 36-layer Qwen3-based transformer (4.13B params) that processes the full token sequence with GQA (32 heads, 8 KV heads), RoPE at 1M base, QK norm, and a persistent KV cache
- **Fast-AR** — a 4-layer transformer (0.42B params) that autoregressively generates 10 acoustic codebook tokens from the Slow-AR hidden state for each semantic step
- **Audio codec** — a convolutional encoder/decoder with residual vector quantization (RVQ, 10 codebooks × 4096 entries) that converts between audio waveforms and discrete codes

Total: ~4.56B parameters.

---

## Implementation notes

The C++ engine (`src/`) is built entirely on [ggml](https://github.com/ggml-org/ggml) (unmodified, pinned as a submodule). Key design decisions:

- **Separate persistent `gallocr` allocators** for Slow-AR and Fast-AR — each path keeps its own compute buffer, avoiding memory re-planning per token
- **Temporary prefill allocator** — freed immediately after prefill, so the large compute buffer does not persist into the generation loop
- **Codec on CPU** — the audio codec executes exactly twice per synthesis (encode reference + decode output), so running it on CPU has zero impact on generation throughput
- **posix_fadvise(DONTNEED)** after loading the weights *(Linux only)* — advises the kernel to drop the GGUF file from page cache once the tensors are already in the backend buffer, reducing duplicate RAM use
- **Correct ByteLevel tokenization** — the GPT-2 byte-to-unicode table is applied before BPE, producing token IDs identical to the HuggingFace reference tokenizer

---

## Tips

### Long outputs

Voice quality and amplitude tend to degrade after ~800 tokens (~37 s of audio). For longer texts, split into sentences and concatenate the resulting WAV files. By default, the engine applies peak normalisation on save to partially compensate, but splitting remains the most reliable approach.

---

## Known limitations (alpha)

- No streaming output — WAV is written only after full generation completes
- No batch inference
- Voice cloning quality depends heavily on reference audio length and SNR
- Windows: CUDA and Vulkan backends are supported; when using MSVC 2019+, ensure CUDA ≥ 12.4 is installed before building
- macOS is untested

---

## License

The model weights and associated materials are licensed under the **Fish Audio Research License**. Key points:

- **Research and non-commercial use:** free, under the terms of this Agreement
- **Commercial use:** requires a separate written license from Fish Audio
- When distributing, you must include a copy of the license and the attribution notice
- Attribution: *"This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved."*

Full license: [LICENSE.md](LICENSE.md)

Commercial licensing: [https://fish.audio](https://fish.audio) · [business@fish.audio](mailto:business@fish.audio)

The inference engine source code (`src/`) is a Derivative Work of the Fish Audio Materials as defined in the Agreement and is distributed under the same Fish Audio Research License terms.
