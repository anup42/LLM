# Gemma 4 26B A4B Runner (Linux, RTX A6000)

This setup runs `google/gemma-4-26B-A4B-it` with an option to use a **pre-downloaded local model folder**.

## 1) Linux setup

```bash
# If python3 is missing:
# Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Install PyTorch for your CUDA/driver compatibility.
# For systems reporting CUDA 12.2 in nvidia-smi, cu121 is usually the safest wheel:
python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

python3 -m pip install -r requirements-gemma4.txt
```

If you previously saw `Could not import module 'AutoProcessor'`, run:

```bash
python3 -m pip install --upgrade --force-reinstall "transformers>=4.57.1,<5" "tokenizers>=0.21.0" "huggingface-hub>=0.31.0"
```

If you saw `list object has no attribute keys`, update to the latest runner code and refresh packages:

```bash
python3 -m pip install --upgrade --force-reinstall -r requirements-gemma4.txt
```

If you see `CUDA GPU not detected by PyTorch`, run:

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, reinstall CUDA-enabled PyTorch (example CUDA 12.4):

```bash
python3 -m pip uninstall -y torch torchvision torchaudio
python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

For CUDA 12.2 driver environments, use:

```bash
python3 -m pip uninstall -y torch torchvision torchaudio
python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## 2) Run using local downloaded model folder (no download)

```bash
python3 gemma4_runner.py \
  --model-path /path/to/gemma-4-26B-A4B-it \
  --local-files-only \
  --prompt "Write a short joke about saving RAM."
```

By default, the script requires CUDA. If you want to force CPU for testing only:

```bash
python3 gemma4_runner.py \
  --model-path /path/to/gemma-4-26B-A4B-it \
  --local-files-only \
  --allow-cpu \
  --prompt "Write a short joke about saving RAM."
```

## 3) Run using Hugging Face model ID

```bash
python3 gemma4_runner.py \
  --model-id google/gemma-4-26B-A4B-it \
  --prompt "Write a short joke about saving RAM."
```

## 4) Dry-run check (no model load/download)

```bash
python3 gemma4_runner.py \
  --model-path /path/to/gemma-4-26B-A4B-it \
  --local-files-only \
  --dry-run
```

## 5) Shortcut script

```bash
chmod +x run_gemma4_linux.sh
./run_gemma4_linux.sh /path/to/gemma-4-26B-A4B-it "Write a short joke about saving RAM."
```

If you pass an empty first argument, it uses the Hugging Face model ID instead of local path:

```bash
./run_gemma4_linux.sh "" "Explain MoE in one sentence."
```
