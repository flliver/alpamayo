# Alpamayo Setup Status

## ✅ Completed

1. **Forked Repository**
   - Fork: https://github.com/flliver/alpamayo
   - Local: `/home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo`

2. **Environment Setup**
   - Virtual environment created (`ar1_venv`)
   - Dependencies installed (without flash-attn, using SDPA)
   - HuggingFace authenticated as: qwert189

3. **Access Granted**
   - ✅ HuggingFace dataset: nvidia/PhysicalAI-Autonomous-Vehicles
   - ✅ HuggingFace model: nvidia/Alpamayo-R1-10B

4. **Demo Script Created**
   - `demo_inference.py` - Uses synthetic data to demonstrate model
   - Currently running: downloading 22GB model weights

## ⚠️ Current Issue

The `physical_ai_av` package (v0.1.0) has a bug when accessing the dataset:

```
File "/home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo/ar1_venv/lib/python3.12/site-packages/physical_ai_av/utils/hf_interface.py", line 231, in download_file
    self.api.get_paths_info(paths=[filename], **self.repo_snapshot_info)[0].size
IndexError: list index out of range
```

This is a bug in the dataset interface library, not an authentication issue.

## 🎯 Current Status

**Running**: `demo_inference.py`
- Creating synthetic camera/egomotion data
- Downloading Alpamayo-R1-10B model (22GB)
- Will demonstrate Chain-of-Causation reasoning with fake data
- Shows model is working even if dataset access is broken

## 🔧 Workarounds

1. **Demo with Synthetic Data** (running now)
   ```bash
   cd /home/ANT.AMAZON.COM/fliver/gt/vla_eval/mayor/rig/alpamayo
   ar1_venv/bin/python demo_inference.py
   ```

2. **Real Data Options** (to try):
   - Wait for `physical_ai_av` package update
   - Download dataset manually and load directly
   - Use the Jupyter notebook (`notebooks/inference.ipynb`)

## 📊 What the Demo Will Show

Once the model download completes (~5-10 min), you'll see:

1. **Chain-of-Causation Reasoning**
   - Natural language explanation of the driving decision
   - Example: "The ego vehicle is approaching an intersection..."

2. **Trajectory Prediction**
   - 64 waypoints over 6.4 seconds
   - XYZ coordinates in ego frame
   - Rotation matrices

3. **Model Architecture**
   - Vision-Language-Action model
   - 10B parameters
   - Running on your RTX 5090 (32GB)

## 📝 Files Created

- `SETUP.md` - Complete setup documentation
- `QUICKSTART.md` - Quick reference
- `run_eval.sh` - Automated runner script
- `check_access.py` - HuggingFace access checker
- `demo_inference.py` - Synthetic data demo (running)
- `STATUS.md` - This file

## Next Steps

1. ✅ Let demo complete to see model working
2. Debug physical_ai_av package or find workaround
3. Get real evaluation with actual driving data

---

Last updated: 2026-03-13 10:35 UTC
