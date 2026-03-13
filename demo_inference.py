#!/usr/bin/env python3
"""Simplified Alpamayo demo with synthetic data (to test model loading and inference)."""

import torch
import numpy as np
from PIL import Image

print("🏔️ Alpamayo Demo Inference")
print("=" * 60)

# Create synthetic multi-camera video data (4 cameras, 4 frames each)
print("\n1. Creating synthetic camera data...")
batch_size = 1
num_cameras = 4
num_frames = 4
height, width = 900, 1600
channels = 3

# Generate random images (in practice, these would be real dashcam frames)
image_frames = torch.randint(0, 255, (num_cameras, num_frames, channels, height, width), dtype=torch.uint8)
print(f"   ✓ Image frames: {image_frames.shape} (cameras, frames, C, H, W)")

# Create camera indices [0, 1, 2, 6] for the 4 cameras
camera_indices = torch.tensor([0, 1, 2, 6], dtype=torch.int64)
print(f"   ✓ Camera indices: {camera_indices.tolist()}")

# Create synthetic egomotion history (16 steps @ 10Hz = 1.6s of history)
print("\n2. Creating synthetic egomotion data...")
num_history_steps = 16
ego_history_xyz = torch.randn(batch_size, 1, num_history_steps, 3) * 0.5  # Small movements
ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, num_history_steps, 1, 1)
print(f"   ✓ Ego history XYZ: {ego_history_xyz.shape}")
print(f"   ✓ Ego history rotation: {ego_history_rot.shape}")

print("\n3. Loading Alpamayo-R1-10B model...")
print("   (This will download 22GB on first run - please wait...)")

try:
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        attn_implementation="eager"  # Use eager mode (SDPA not supported for this model)
    ).to("cuda")
    print("   ✓ Model loaded successfully!")

    processor = helper.get_processor(model.tokenizer)
    print("   ✓ Processor initialized")

    # Prepare model inputs
    print("\n4. Preparing model inputs...")
    messages = helper.create_message(image_frames.flatten(0, 1))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    print("   ✓ Inputs prepared")

    # Run inference
    print("\n5. Running inference...")
    print("   (Generating trajectory predictions with Chain-of-Causation reasoning...)")

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    print("   ✓ Inference complete!")

    # Display results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)

    print("\n🧠 Chain-of-Causation (Reasoning):")
    print("-" * 60)
    for i, cot in enumerate(extra["cot"][0]):
        print(f"\nTrajectory {i+1}:")
        print(cot)

    print("\n" + "-" * 60)
    print(f"\n🎯 Predicted Trajectory:")
    print(f"   Shape: {pred_xyz.shape}")
    print(f"   Predicted {pred_xyz.shape[2]} waypoints over 6.4 seconds (10Hz)")

    # Show first few waypoints
    waypoints = pred_xyz[0, 0, 0, :5, :2].cpu().numpy()  # First 5 waypoints, XY only
    print(f"\n   First 5 waypoints (X, Y in meters):")
    for i, wp in enumerate(waypoints):
        print(f"   t={0.1*(i+1):.1f}s: [{float(wp[0]):6.2f}, {float(wp[1]):6.2f}]")

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("\nNote: This used synthetic data. For real evaluation:")
    print("  1. Ensure HuggingFace access to PhysicalAI-Autonomous-Vehicles dataset")
    print("  2. Run: python src/alpamayo_r1/test_inference.py")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
