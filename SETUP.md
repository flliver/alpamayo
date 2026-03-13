# Alpamayo Setup & Eval Guide

## What is Alpamayo?

Alpamayo 1 is NVIDIA's Vision-Language-Action (VLA) model for autonomous driving. It combines:
- **Chain-of-Causation (CoC) reasoning** - Natural language explanations of driving decisions
- **Trajectory prediction** - 6.4s horizon, 64 waypoints at 10 Hz
- **VLA architecture** - Cosmos-Reason backbone + action expert

**Model**: 10B parameters (Alpamayo-R1-10B)
**Requirements**: NVIDIA GPU with ≥24 GB VRAM (✓ RTX 5090 32GB)

## Setup Completed

### 1. Forked Repository
- **Original**: https://github.com/NVlabs/alpamayo
- **Fork**: https://github.com/flliver/alpamayo
- **Location**: `/home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo`

### 2. Environment Setup
- Created virtual environment: `ar1_venv`
- Installed dependencies (without flash-attn - see modifications below)
- Using uv package manager (0.10.9)

### 3. Modifications Made

**Problem**: flash-attn requires CUDA compiler (nvcc) which isn't installed.

**Solution**: Removed flash-attn and configured model to use PyTorch's SDPA (Scaled Dot-Product Attention).

**Changed files**:
- `pyproject.toml` - Removed `flash-attn>=2.8.3` dependency
- `src/alpamayo_r1/test_inference.py` - Added `attn_implementation="sdpa"` to model loading

This is officially supported per the README troubleshooting section.

## HuggingFace Authentication (TODO)

The model and dataset are gated resources requiring access approval:

1. **Request Access** (first-time only):
   - Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
   - Model: https://huggingface.co/nvidia/Alpamayo-R1-10B

2. **Get Access Token**:
   - Visit: https://huggingface.co/settings/tokens
   - Create a new token with read permissions

3. **Authenticate**:
   ```bash
   cd /home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo
   source ar1_venv/bin/activate
   hf auth login
   # Paste your token when prompted
   ```

## Running the Basic Eval

Once authenticated, run the test inference script:

```bash
cd /home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo
source ar1_venv/bin/activate
python src/alpamayo_r1/test_inference.py
```

### What the Eval Does

1. **Loads a test clip** - Clip ID: `030c760c-ae38-49aa-9ad8-f5650a545d26` at t0=5.1s
2. **Loads the model** - Downloads 22GB model weights (first run only, ~2.5 min on 100 MB/s)
3. **Runs inference** - Generates trajectory predictions and reasoning traces
4. **Computes minADE** - Minimum Average Displacement Error (meters)
   - Measures how close the predicted trajectory is to ground truth
   - Lower is better

### Expected Output

```
Loading dataset for clip_id: 030c760c-ae38-49aa-9ad8-f5650a545d26...
Dataset loaded.
[Model loading output...]
Chain-of-Causation (per trajectory):
[Reasoning trace explaining the driving decision]
minADE: X.XX meters
Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling...
```

### Key Parameters

In `test_inference.py` line 60:
- `num_traj_samples=1` - Number of trajectories to generate (increase for more samples)
- `temperature=0.6` - Sampling temperature
- `top_p=0.98` - Nucleus sampling threshold

## Interactive Notebook

For visual sanity checks and more detailed exploration:
```bash
jupyter notebook notebooks/inference.ipynb
```

## Project Structure

```
alpamayo/
├── src/alpamayo_r1/
│   ├── test_inference.py    # Main eval script
│   ├── models/              # Model components
│   ├── diffusion/           # Diffusion model components
│   └── ...
├── notebooks/
│   └── inference.ipynb      # Interactive inference
├── pyproject.toml           # Dependencies (modified)
└── ar1_venv/                # Virtual environment
```

## Troubleshooting

### CUDA Out-of-Memory Errors
- Ensure GPU has ≥24GB VRAM (check with `nvidia-smi`)
- Reduce `num_traj_samples` if generating multiple trajectories
- Close other GPU-intensive applications

### Download Times
- Model weights: 22GB (~2.5 min on 100 MB/s connection)
- Dataset samples: Relatively small, downloads on first use

### Model Outputs
- Outputs are nondeterministic due to sampling
- With `num_traj_samples=1`, expect variance in minADE
- Increase samples for more stable results (uses more VRAM)

## License

- **Code**: Apache License 2.0
- **Model Weights**: Non-commercial license (research/evaluation only)

## References

- **Paper**: [Alpamayo-R1: Bridging Reasoning and Action Prediction](https://arxiv.org/abs/2511.00088)
- **HuggingFace Model Card**: https://huggingface.co/nvidia/Alpamayo-R1-10B
- **NVIDIA News**: https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development
