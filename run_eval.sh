#!/bin/bash
# Quick script to run Alpamayo basic eval

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🏔️  Alpamayo Basic Eval Runner"
echo "================================"
echo ""

# Check if venv exists
if [ ! -d "ar1_venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   uv venv ar1_venv"
    echo "   source ar1_venv/bin/activate"
    echo "   uv sync --active"
    exit 1
fi

# Activate venv
echo "📦 Activating virtual environment..."
source ar1_venv/bin/activate

# Check HuggingFace auth
echo "🔐 Checking HuggingFace authentication..."
if ! hf auth whoami &>/dev/null; then
    echo "❌ Not logged into HuggingFace. Please authenticate:"
    echo "   hf auth login"
    echo ""
    echo "   Then request access to:"
    echo "   - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles"
    echo "   - https://huggingface.co/nvidia/Alpamayo-R1-10B"
    exit 1
fi

HF_USER=$(hf auth whoami 2>/dev/null | grep -oP 'username: \K.*' || echo "authenticated")
echo "✓ Logged in as: $HF_USER"
echo ""

# Check GPU
echo "🎮 Checking GPU availability..."
if ! nvidia-smi &>/dev/null; then
    echo "❌ NVIDIA GPU not detected. This model requires a GPU with ≥24GB VRAM."
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "✓ GPU detected: $GPU_INFO"
echo ""

# Run inference
echo "🚀 Running inference..."
echo "   (First run will download model weights - 22GB, ~2.5 min on 100 MB/s)"
echo ""

python src/alpamayo_r1/test_inference.py

echo ""
echo "✅ Eval complete!"
