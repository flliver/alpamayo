# Alpamayo Quick Start

## Ready to Run? (After HuggingFace Auth)

```bash
cd /home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo
./run_eval.sh
```

## First Time Setup

### 1. HuggingFace Authentication

```bash
# Request access first (one-time):
# - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
# - https://huggingface.co/nvidia/Alpamayo-R1-10B

# Then login:
cd /home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo
source ar1_venv/bin/activate
hf auth login
# Paste your token from https://huggingface.co/settings/tokens
```

### 2. Run Eval

```bash
./run_eval.sh
```

Or manually:
```bash
cd /home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo
source ar1_venv/bin/activate
python src/alpamayo_r1/test_inference.py
```

## What You'll Get

- **Chain-of-Causation reasoning** - Natural language explanation of driving decisions
- **minADE metric** - Average trajectory error in meters (lower is better)
- **First run**: Downloads 22GB model weights (~2.5 min on fast connection)

## Customization

Edit `src/alpamayo_r1/test_inference.py` line 60:
```python
num_traj_samples=1,  # Increase for more trajectory samples (uses more VRAM)
```

## Full Documentation

See `SETUP.md` for complete details, troubleshooting, and architecture info.
