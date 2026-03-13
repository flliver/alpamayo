#!/usr/bin/env python3
"""Test inference with real dataset (with physical_ai_av bug fix)."""

import torch
import numpy as np

# IMPORTANT: Apply bug fix BEFORE importing anything else
print("Applying physical_ai_av bug fix...")
import fix_physical_ai_av

# Now import the rest
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

print("=" * 60)
print("🏔️ Alpamayo Real Dataset Test")
print("=" * 60)

# Example clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"\nLoading dataset for clip_id: {clip_id}...")
print("(First run will download data - this may take a while)")

try:
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print("✓ Dataset loaded successfully!")
    print(f"  Image frames shape: {data['image_frames'].shape}")
    print(f"  Camera indices: {data['camera_indices'].tolist()}")
    print(f"  Ego history shape: {data['ego_history_xyz'].shape}")

except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    import traceback
    traceback.print_exc()
    print("\nThis might be due to:")
    print("  1. Large data download required")
    print("  2. Network issues")
    print("  3. Additional package bugs")
    exit(1)

print("\nPreparing model inputs...")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

print("Loading Alpamayo-R1-10B model...")
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=torch.bfloat16,
    attn_implementation="eager"
).to("cuda")

processor = helper.get_processor(model.tokenizer)

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
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

model_inputs = helper.to_device(model_inputs, "cuda")

print("Running inference...")
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

# Display results
print("\n" + "=" * 60)
print("📊 RESULTS (Real Dataset)")
print("=" * 60)

print("\n🧠 Chain-of-Causation:")
print("-" * 60)
for i, cot in enumerate(extra["cot"][0]):
    print(f"Trajectory {i+1}: {cot}")

# Calculate minADE with ground truth
gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()

print("\n" + "-" * 60)
print(f"\n🎯 Trajectory Accuracy:")
print(f"   minADE: {min_ade:.3f} meters")
print(f"   (Average displacement error - lower is better)")

print("\n   Predicted trajectory shape: {pred_xyz.shape}")
print(f"   Ground truth shape: {gt_xy.shape}")

# Show first few waypoints
waypoints = pred_xyz[0, 0, 0, :5, :2].cpu().numpy()
print(f"\n   First 5 predicted waypoints (X, Y in meters):")
for i, wp in enumerate(waypoints):
    gt_wp = gt_xy[:, i]
    error = np.linalg.norm(wp - gt_wp)
    print(f"   t={0.1*(i+1):.1f}s: pred=[{float(wp[0]):6.2f}, {float(wp[1]):6.2f}] "
          f"gt=[{gt_wp[0]:6.2f}, {gt_wp[1]:6.2f}] error={error:.2f}m")

print("\n" + "=" * 60)
print("✅ Real dataset test complete!")
print("=" * 60)
