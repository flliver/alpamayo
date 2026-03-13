#!/usr/bin/env python3
"""Check HuggingFace access to Alpamayo resources."""

from huggingface_hub import HfApi
import sys

api = HfApi()

print("🔐 Checking HuggingFace Access")
print("=" * 50)

# Check dataset access
print("\n1. Dataset: nvidia/PhysicalAI-Autonomous-Vehicles")
try:
    dataset_info = api.dataset_info("nvidia/PhysicalAI-Autonomous-Vehicles")
    print("   ✅ Access granted!")
except Exception as e:
    if "403" in str(e) or "gated" in str(e).lower():
        print("   ❌ Access needed - visit:")
        print("      https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles")
    else:
        print(f"   ❌ Error: {e}")

# Check model access
print("\n2. Model: nvidia/Alpamayo-R1-10B")
try:
    model_info = api.model_info("nvidia/Alpamayo-R1-10B")
    print("   ✅ Access granted!")
except Exception as e:
    if "403" in str(e) or "gated" in str(e).lower():
        print("   ❌ Access needed - visit:")
        print("      https://huggingface.co/nvidia/Alpamayo-R1-10B")
    else:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 50)
print("Once both show ✅, you can run the eval with:")
print("  ./run_eval.sh")
