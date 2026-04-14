# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n omnivla python=3.10 -y
conda activate omnivla

# Install PyTorch (CPU version used in this setup)
pip install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Clone repo and install dependencies
git clone https://github.com/NHirose/OmniVLA.git
cd OmniVLA
pip install -e .

# Optional: install Flash Attention 2 for training (GPU/CUDA env)
# https://github.com/Dao-AILab/flash-attention
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Download Inference Checkpoints

```bash
# OmniVLA main checkpoint (required by inference/run_omnivla.py)
git clone https://huggingface.co/NHirose/omnivla-original

# OmniVLA-edge checkpoint (required by inference/run_omnivla_edge.py)
git clone https://huggingface.co/NHirose/omnivla-edge
```

## Run Inference

```bash
conda activate omnivla
cd OmniVLA

# OmniVLA
python inference/run_omnivla.py

# OmniVLA-edge
python inference/run_omnivla_edge.py
```

Expected output images:

- `inference/1_ex.jpg`
- `inference/1_ex_omnivla_edge.jpg`

## Notes

- `run_omnivla.py` and `run_omnivla_edge.py` support CPU fallback.
- If `run_omnivla_edge.py` reports missing `pkg_resources`, install:

```bash
pip install "setuptools==80.9.0"
```
