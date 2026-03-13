# Alpamayo R1 - Usage Guide

## ⚡ Fastest Start (Uses Existing Setup)

```bash
make demo-quick
```

That's it! The model is already downloaded and configured.

## 📦 Full Setup (Fresh Install)

### Prerequisites

```bash
# On Ubuntu/Debian - install Python 3.12 venv support
sudo apt install python3.12-venv
```

### Install & Run

```bash
make venv         # Create virtual environment
make install      # Install dependencies
make demo         # Run demo
```

## 🎯 Makefile Commands

```bash
make help         # Show all commands
make demo-quick   # Run demo with existing ar1_venv (FASTEST)
make demo         # Run demo with new venv
make venv         # Create virtual environment
make install      # Install dependencies
make install-dev  # Install with dev tools (Jupyter)
make check-hf     # Check HuggingFace access
make run-eval     # Run eval with real dataset (blocked by bug)
make clean        # Remove venv and cache
```

## 📊 What You'll See

```
🏔️ Alpamayo Demo Inference
============================================================

🧠 Chain-of-Causation (Reasoning):
------------------------------------------------------------
['Keep lane to continue driving since no critical agent needs attention.']

🎯 Predicted Trajectory:
   Shape: torch.Size([1, 1, 1, 64, 3])
   Predicted 64 waypoints over 6.4 seconds (10Hz)

   First 5 waypoints (X, Y in meters):
   t=0.1s: [  1.36,   0.00]
   t=0.2s: [  2.71,   0.00]
   t=0.3s: [  4.06,   0.01]
   t=0.4s: [  5.40,   0.02]
   t=0.5s: [  6.74,   0.02]
```

## 🔧 Manual Usage

### Easy Way (Recommended)
```bash
source activate.sh
python demo_inference.py
```

The `activate.sh` script activates the venv AND sets PYTHONPATH automatically.

### Manual Way
```bash
# Using venv
source venv/bin/activate
export PYTHONPATH=/home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo/src:$PYTHONPATH
python demo_inference.py

# Or using ar1_venv
export PATH="$HOME/.local/bin:$PATH"
source ar1_venv/bin/activate
export PYTHONPATH=/home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo/src:$PYTHONPATH
python demo_inference.py
```

## 📁 Project Structure

```
alpamayo/
├── Makefile                # Build automation ⭐
├── requirements.txt        # Traditional pip dependencies
├── pyproject.toml          # Project metadata (uv-based)
├── demo_inference.py       # Demo with synthetic data ⭐
├── run_eval.sh            # Automated eval runner
├── check_access.py        # HF access checker
├── ar1_venv/              # Existing venv (uv-based)
├── venv/                  # New venv (pip-based, optional)
├── SETUP.md               # Detailed setup guide
├── QUICKSTART.md          # Quick reference
└── src/alpamayo_r1/
    └── test_inference.py  # Real eval (blocked by dataset bug)
```

## ⚙️ Requirements

- **Python**: 3.12 (exact version required by project)
- **GPU**: NVIDIA with ≥24GB VRAM (RTX 3090/4090, A5000, H100)
- **CUDA**: Compatible PyTorch (auto-installed)
- **Disk**: ~22GB for model weights (already downloaded)
- **OS**: Linux (tested on Ubuntu)

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'" or "No module named 'alpamayo_r1'"
→ Use `make demo` or `make demo-quick` (handles PYTHONPATH automatically)
→ Or use helper: `source activate.sh` then run your script
→ Or set manually: `export PYTHONPATH=$(pwd)/src:$PYTHONPATH`

### "CUDA out of memory"
→ GPU needs ≥24GB VRAM
→ Close other GPU applications
→ Check: `nvidia-smi`

### "physical_ai_av" dataset error
→ Known bug in dataset package (IndexError)
→ Demo uses synthetic data (works fine)
→ Real dataset eval currently blocked

### Python version mismatch
→ Project requires Python ==3.12.*
→ Your default python3 may be different
→ Use: `make venv` (uses python3.12 explicitly)

## 🎓 Two Setup Approaches

### 1. ar1_venv (Existing, uv-based)
- **Pros**: Already set up, faster
- **Cons**: Requires uv tool
- **Use**: `make demo-quick`

### 2. venv (New, pip-based)
- **Pros**: Standard Python, more portable
- **Cons**: Requires python3.12-venv package
- **Use**: `make venv && make install && make demo`

Both work identically - use whichever you prefer!

## 📖 More Info

- **SETUP.md** - Complete setup documentation
- **QUICKSTART.md** - One-page quick reference
- **STATUS.md** - Current project status
- **Original README** - See upstream NVlabs/alpamayo
