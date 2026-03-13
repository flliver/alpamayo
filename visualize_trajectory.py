#!/usr/bin/env python3
"""
Visualize predicted and ground truth trajectories overlaid on video frames.

This script:
1. Loads a clip from the PhysicalAI-AV dataset
2. Runs Alpamayo-R1 inference
3. Renders predicted and ground truth trajectories on video frames
4. Saves the result as a video file
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
import mediapy as mp
from pathlib import Path

from custom_dataset_loader import load_clip_data
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


def rotate_90cc(xy):
    """Rotate (x, y) by 90 deg CCW -> (y, -x) for bird's eye view."""
    return np.stack([-xy[1], xy[0]], axis=0)


def create_trajectory_overlay(pred_xyz, gt_xyz, frame_width=400, frame_height=400):
    """
    Create a bird's eye view trajectory plot as an image.

    Args:
        pred_xyz: Predicted trajectory [num_samples, num_steps, 3]
        gt_xyz: Ground truth trajectory [num_steps, 3]
        frame_width: Width of output image in pixels
        frame_height: Height of output image in pixels

    Returns:
        numpy array of shape (H, W, 3) with RGB values [0, 255]
    """
    dpi = 100
    fig = plt.figure(figsize=(frame_width / dpi, frame_height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    # Plot ground truth
    gt_xy = gt_xyz[:, :2].T
    gt_xy_rot = rotate_90cc(gt_xy)
    ax.plot(*gt_xy_rot, 'r-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax.plot(*gt_xy_rot, 'ro', markersize=3)

    # Plot predictions (multiple samples if available)
    for i in range(pred_xyz.shape[0]):
        pred_xy = pred_xyz[i, :, :2].T
        pred_xy_rot = rotate_90cc(pred_xy)
        label = f'Predicted #{i+1}' if pred_xyz.shape[0] > 1 else 'Predicted'
        ax.plot(*pred_xy_rot, 'b-', linewidth=2, label=label, alpha=0.7)
        ax.plot(*pred_xy_rot, 'bo', markersize=2)

    # Mark ego position (origin)
    ax.plot(0, 0, 'g*', markersize=15, label='Ego Vehicle', zorder=10)

    ax.set_xlabel('Lateral (meters)', fontsize=10)
    ax.set_ylabel('Forward (meters)', fontsize=10)
    ax.set_title('Trajectory Prediction (Bird\'s Eye View)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Set reasonable view bounds
    all_xy = np.concatenate([
        gt_xy_rot.T,
        pred_xy_rot.T,
    ], axis=0)
    margin = 5  # meters
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]  # Drop alpha channel

    plt.close(fig)
    return img


def overlay_on_frame(frame, trajectory_img, position='bottom-right', scale=0.3):
    """
    Overlay trajectory visualization on a video frame.

    Args:
        frame: Video frame [H, W, 3] uint8
        trajectory_img: Trajectory visualization [H2, W2, 3] uint8
        position: Where to place overlay ('bottom-right', 'bottom-left', 'top-right', 'top-left')
        scale: Scale factor for overlay size

    Returns:
        Composited frame with trajectory overlay
    """
    frame = frame.copy()

    # Resize trajectory image
    traj_h, traj_w = trajectory_img.shape[:2]
    new_h = int(traj_h * scale)
    new_w = int(traj_w * scale)

    import cv2
    traj_small = cv2.resize(trajectory_img, (new_w, new_h))

    # Determine position
    frame_h, frame_w = frame.shape[:2]
    margin = 10

    if position == 'bottom-right':
        y1 = frame_h - new_h - margin
        x1 = frame_w - new_w - margin
    elif position == 'bottom-left':
        y1 = frame_h - new_h - margin
        x1 = margin
    elif position == 'top-right':
        y1 = margin
        x1 = frame_w - new_w - margin
    else:  # top-left
        y1 = margin
        x1 = margin

    y2 = y1 + new_h
    x2 = x1 + new_w

    # Add semi-transparent background
    alpha = 0.9
    frame[y1:y2, x1:x2] = (
        frame[y1:y2, x1:x2] * (1 - alpha) +
        traj_small * alpha
    ).astype(np.uint8)

    return frame


def add_text_overlay(frame, text, position='top-left'):
    """Add text overlay to frame."""
    import cv2
    frame = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    margin = 10
    padding = 5

    if position == 'top-left':
        x = margin
        y = margin + text_h + padding
    elif position == 'top-center':
        x = (frame.shape[1] - text_w) // 2
        y = margin + text_h + padding
    else:  # top-right
        x = frame.shape[1] - text_w - margin
        y = margin + text_h + padding

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1
    )

    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def visualize_clip_with_trajectories(
    clip_id,
    output_path="trajectory_visualization.mp4",
    t0_us=None,
    num_history_steps=16,
    num_future_steps=64,
    num_frames=16,
    camera_name="camera_front_wide_120fov",
    num_traj_samples=1,
    fps=10,
):
    """
    Create a video with predicted and ground truth trajectories.

    Args:
        clip_id: Clip ID to visualize
        output_path: Where to save the output video
        t0_us: Reference time in microseconds (auto-selected if None)
        num_history_steps: Number of history steps for model input
        num_future_steps: Number of future steps to predict
        num_frames: Number of camera frames to extract
        camera_name: Which camera to visualize
        num_traj_samples: Number of trajectory samples to generate
        fps: Output video frame rate
    """
    print("=" * 70)
    print("TRAJECTORY VISUALIZATION")
    print("=" * 70)

    # Load model
    print("\n1. Loading Alpamayo-R1 model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("   Model loaded!")

    # Load clip data
    print(f"\n2. Loading clip: {clip_id}")
    data = load_clip_data(
        clip_id=clip_id,
        t0_us=t0_us,
        num_history_steps=num_history_steps,
        num_future_steps=num_future_steps,
        num_frames=4,  # For model input
        camera_names=[camera_name],
    )
    print("   Clip data loaded!")

    # Run inference
    print("\n3. Running model inference...")
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
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

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=256,
            return_extra=True,
        )

    cot = extra["cot"][0]
    print(f"   Chain-of-Causation: {cot}")

    # Calculate accuracy
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print(f"   minADE: {min_ade:.3f} meters")

    # Save trajectory data before loading more frames
    pred_xyz_np = pred_xyz.cpu().numpy()[0, 0]  # [num_samples, num_steps, 3]
    gt_xyz_np = data["ego_future_xyz"].cpu().numpy()[0, 0]  # [num_steps, 3]
    inference_t0 = data["absolute_timestamps"][0, 0].item()

    # Load more frames for video
    print(f"\n4. Loading {num_frames} video frames from {camera_name}...")
    video_data = load_clip_data(
        clip_id=clip_id,
        t0_us=inference_t0,  # Use same t0 as inference
        num_history_steps=num_history_steps,
        num_future_steps=num_future_steps,
        num_frames=num_frames,
        camera_names=[camera_name],
    )

    frames = video_data["image_frames"][0]  # [num_frames, 3, H, W]
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [num_frames, H, W, 3]
    frames = (frames * 255).astype(np.uint8)
    print(f"   Loaded {len(frames)} frames, shape: {frames[0].shape}")

    # Create trajectory overlay
    print("\n5. Creating trajectory overlay...")

    traj_img = create_trajectory_overlay(pred_xyz_np, gt_xyz_np)

    # Compose frames with overlay
    print("\n6. Compositing frames...")
    output_frames = []
    for i, frame in enumerate(frames):
        # Add trajectory overlay
        composed = overlay_on_frame(frame, traj_img, position='bottom-right', scale=0.35)

        # Add text overlays
        timestamp = video_data["absolute_timestamps"][0, i].item() / 1e6  # Convert to seconds
        composed = add_text_overlay(
            composed,
            f"t = {timestamp:.2f}s | minADE: {min_ade:.2f}m",
            position='top-left'
        )

        # Add reasoning text (split into multiple lines if needed)
        if len(cot[0]) < 80:
            reasoning_text = cot[0]
        else:
            reasoning_text = cot[0][:77] + "..."

        composed = add_text_overlay(
            composed,
            f"Reasoning: {reasoning_text}",
            position='top-center'
        )

        output_frames.append(composed)

    # Save video (try mediapy, fallback to opencv if ffmpeg not available)
    print(f"\n7. Saving video to {output_path}...")
    try:
        # Convert RGB to BGR for video encoding
        import cv2
        output_frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in output_frames]
        mp.write_video(output_path, output_frames_bgr, fps=fps)
        print(f"   Video saved with mediapy! ({len(output_frames)} frames @ {fps} fps)")
    except RuntimeError as e:
        if "ffmpeg" in str(e):
            print("   ffmpeg not found, using opencv instead...")
            import cv2
            height, width = output_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frame in output_frames:
                # Convert RGB to BGR for opencv
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"   Video saved with opencv! ({len(output_frames)} frames @ {fps} fps)")
        else:
            raise

    # Also save a single comparison image
    comparison_path = output_path.replace('.mp4', '_comparison.png')
    comparison_frame = output_frames[len(output_frames) // 2]  # Middle frame
    mp.write_image(comparison_path, comparison_frame)
    print(f"   Comparison image saved: {comparison_path}")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"Output video: {output_path}")
    print(f"Comparison image: {comparison_path}")
    print(f"Clip: {clip_id}")
    print(f"Camera: {camera_name}")
    print(f"Reasoning: {cot[0]}")
    print(f"minADE: {min_ade:.3f} meters")

    return output_path


if __name__ == "__main__":
    import sys

    # Default test clip
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"

    # Allow command line override
    if len(sys.argv) > 1:
        clip_id = sys.argv[1]

    output_path = f"trajectory_viz_{clip_id[:8]}.mp4"

    visualize_clip_with_trajectories(
        clip_id=clip_id,
        output_path=output_path,
        t0_us=5_100_000,  # 5.1 seconds into the clip
        num_frames=32,  # 3.2 seconds of video @ 10fps
        camera_name="camera_front_wide_120fov",
        num_traj_samples=1,
        fps=10,
    )
