# AI Captioning — Streamlit UI (microsoft/Florence-2-base on CPU)

A simple image captioning application using Microsoft / Florence-2-base run locally on CPU with a Streamlit user interface. This README explains environment setup, required libraries, how to download or access the model, an example Streamlit usage snippet, performance tips for CPU-only inference, and troubleshooting notes.

This project uses:
- Python virtual environment (venv)
- Streamlit for the UI
- PyTorch (CPU)
- Hugging Face Transformers
- Accelerate (optional for nicer device/config handling)
- Pillow for image IO
- huggingface-hub for model access

Important: this README assumes you will run the model on CPU. Florence-2-base is a large model and CPU inference can be slow — read the Performance tips section below.

---

## Features

- Upload an image in the browser and generate a natural-language caption
- Lightweight Streamlit UI for quick demos
- Uses Hugging Face model "microsoft/Florence-2-base" as the captioning backbone
- Designed to run on CPU-only machines

---

## Recommended project layout

- app.py                - Streamlit app entrypoint (example snippet below)
- requirements.txt      - Python dependencies
- .env                  - optional environment variables (HUGGINGFACE_TOKEN, etc.)
- README.md

---

## Prerequisites

- Python 3.8+
- Git (optional, to clone model/code)
- Enough disk space to cache the model (check model card for size)
- Internet access for initial model download (unless you pre-download the model)

If the model is gated or requires authentication, you will need a Hugging Face token (see next section).

---

## Getting a Hugging Face token (if needed)

If the model requires access control, authenticate to Hugging Face:

1. Create a token at https://huggingface.co/settings/tokens
2. On your machine run:
   ```
   huggingface-cli login
   ```
   or set the token in an environment variable:
   ```
   export HUGGINGFACE_HUB_TOKEN="hf_..."
   ```

---

## Create and activate a virtual environment

Linux / macOS
```
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## Install dependencies

1. Minimal requirements (example requirements.txt)
```
streamlit
transformers
accelerate
pillow
huggingface-hub
# torch must be installed for CPU builds from the official PyTorch index:
# See https://pytorch.org for up-to-date instructions for your platform.
```

2. Recommended install sequence (ensures CPU wheels for torch):
```
# Install CPU PyTorch wheels (this index URL installs CPU-only builds)
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install the rest
pip install streamlit transformers accelerate pillow huggingface-hub
```

Note: If you are on a platform that provides a different recommended torch command (Windows / special Python versions), follow the instructions at https://pytorch.org.

---

## Model: microsoft/Florence-2-base

- Model identifier: microsoft/Florence-2-base
- Hosted on Hugging Face: https://huggingface.co/microsoft/Florence-2-base (check the model card)
- Confirm the model supports the "image-to-text" or captioning pipeline — consult the model card for the recommended usage class and any special processors.

First run will download weights to your local Hugging Face cache (~/.cache/huggingface). Download can be large; be patient.

---

---

## Configuration & environment variables

You may want to set:
- HUGGINGFACE_HUB_TOKEN (or use huggingface-cli login)
- TRANSFORMERS_CACHE to a custom cache path
- OMP_NUM_THREADS / MKL_NUM_THREADS to limit CPU thread usage

Example:
```
export HUGGINGFACE_HUB_TOKEN="hf_..."
export TRANSFORMERS_CACHE="/path/to/cache"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

Limiting threads often helps with system responsiveness when running heavy CPU inference.

---

## CPU performance tips

- Expect higher latency on CPU vs GPU. Captioning a single image can take many seconds to minutes depending on CPU capabilities.
- Restrict the number of threads:
  - Set environment variables OMP_NUM_THREADS and MKL_NUM_THREADS before running to limit parallel BLAS/OpenMP threads.
  - In Python you can also set torch.set_num_threads(n) after importing torch.
- Use `torch.no_grad()` and `model.eval()` (the pipeline often handles this internally).
- Use deterministic generation settings (do_sample=False, temperature=0.0) to reduce variability and some overhead.
- If you need lower latency and can accept lower quality, consider using a smaller model or offloading to a machine with a GPU.
- Consider batched processing if you have many images to caption (still check memory constraints).

---


## Acknowledgements & references

- Hugging Face Transformers: https://github.com/huggingface/transformers
- Microsoft Florence model card: https://huggingface.co/microsoft/Florence-2-base
- Streamlit: https://streamlit.io
- PyTorch installation guide: https://pytorch.org

---
!!! The Model can Hallucinate because I have used a base model which cannot provide the accuracy.
  You can use "large" instead of "Base" for better performance.
