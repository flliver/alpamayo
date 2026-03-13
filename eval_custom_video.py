#!/usr/bin/env python3
"""
Evaluate Alpamayo-R1 on custom video files.

This script:
1. Loads a custom video file (e.g., dashcam footage)
2. Extracts frames and formats them for the model
3. Uses synthetic egomotion data (since real pose data is unavailable)
4. Runs Alpamayo-R1 inference to predict trajectories
5. Visualizes predicted trajectories overlaid on the video

Note: Without real egomotion/pose data, the model's predictions may be less accurate,
but this demonstrates the inference pipeline on arbitrary videos.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
from einops import rearrange

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


def extract_frames_from_video(video_path, num_frames=4, target_fps=10):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_fps: Target framerate for extraction (default: 10 Hz to match model training)

    Returns:
        frames: numpy array of shape (num_frames, H, W, 3) uint8 RGB
        fps: original video FPS
    """
    print(f"\nExtracting {num_frames} frames from video...")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video info: {width}x{height}, {fps:.2f} FPS, {duration:.2f}s ({total_frames} frames)")

    # Calculate frame indices to extract at target_fps
    # We want num_frames spaced at target_fps intervals
    frame_interval = int(fps / target_fps) if fps > target_fps else 1

    # Start from a reasonable point (skip first 1 second if video is long enough)
    start_frame = int(fps) if total_frames > (num_frames * frame_interval + fps) else 0

    frames = []
    frame_indices = []

    for i in range(num_frames):
        frame_idx = start_frame + i * frame_interval
        if frame_idx >= total_frames:
            print(f"  Warning: Requested frame {frame_idx} exceeds video length, wrapping around")
            frame_idx = frame_idx % total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_indices.append(frame_idx)

    cap.release()

    print(f"  Extracted frames at indices: {frame_indices}")
    print(f"  Frame shape: {frames[0].shape}")

    return np.array(frames), fps


def create_synthetic_egomotion(num_history_steps=16, motion_type="stationary"):
    """
    Create synthetic egomotion data for model input.

    Since we don't have real pose measurements, we'll create plausible synthetic data.

    Args:
        num_history_steps: Number of history steps (default: 16)
        motion_type: Type of motion to simulate ("stationary", "forward", "curve_left", "curve_right")

    Returns:
        ego_history_xyz: torch.Tensor (1, 1, num_history_steps, 3)
        ego_history_rot: torch.Tensor (1, 1, num_history_steps, 3, 3)
    """
    print(f"\nCreating synthetic egomotion ({motion_type})...")

    if motion_type == "stationary":
        # Vehicle standing still
        xyz = torch.zeros(1, 1, num_history_steps, 3)

    elif motion_type == "forward":
        # Moving forward at ~10 m/s (36 km/h)
        # History goes backward in time, so positions should be negative
        time_step = 0.1  # 10 Hz
        velocity = -10.0  # meters per second (negative because history)
        positions = np.array([i * time_step * velocity for i in range(num_history_steps)])
        xyz = torch.zeros(1, 1, num_history_steps, 3)
        xyz[0, 0, :, 0] = torch.tensor(positions[::-1].copy())  # X axis (forward)

    elif motion_type == "curve_left":
        # Following a left curve
        time_step = 0.1
        velocity = 10.0
        radius = 50.0  # meters
        angles = np.linspace(0, num_history_steps * time_step * velocity / radius, num_history_steps)
        x = radius * np.sin(angles)
        y = radius * (1 - np.cos(angles))
        xyz = torch.zeros(1, 1, num_history_steps, 3)
        xyz[0, 0, :, 0] = torch.tensor(x[::-1].copy())
        xyz[0, 0, :, 1] = torch.tensor(y[::-1].copy())

    elif motion_type == "curve_right":
        # Following a right curve
        time_step = 0.1
        velocity = 10.0
        radius = 50.0
        angles = np.linspace(0, num_history_steps * time_step * velocity / radius, num_history_steps)
        x = radius * np.sin(angles)
        y = -radius * (1 - np.cos(angles))
        xyz = torch.zeros(1, 1, num_history_steps, 3)
        xyz[0, 0, :, 0] = torch.tensor(x[::-1].copy())
        xyz[0, 0, :, 1] = torch.tensor(y[::-1].copy())

    else:
        raise ValueError(f"Unknown motion_type: {motion_type}")

    # Identity rotation (no turning)
    rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, num_history_steps, 1, 1)

    print(f"  History XYZ: {xyz.shape}")
    print(f"  History rotation: {rot.shape}")

    return xyz, rot


def rotate_90cc(xy):
    """Rotate (x, y) by 90 deg CCW for bird's eye view."""
    return np.stack([-xy[1], xy[0]], axis=0)


def create_trajectory_plot(pred_xyz_np, frame_width=400, frame_height=400):
    """
    Create a bird's eye view trajectory plot.

    Args:
        pred_xyz_np: Predicted trajectory [num_samples, num_steps, 3]
        frame_width: Width of output image
        frame_height: Height of output image

    Returns:
        numpy array of shape (H, W, 3) with RGB values [0, 255]
    """
    dpi = 100
    fig = plt.figure(figsize=(frame_width / dpi, frame_height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    # Plot predictions (multiple samples if available)
    for i in range(pred_xyz_np.shape[0]):
        pred_xy = pred_xyz_np[i, :, :2].T
        pred_xy_rot = rotate_90cc(pred_xy)
        label = f'Predicted #{i+1}' if pred_xyz_np.shape[0] > 1 else 'Predicted'
        ax.plot(*pred_xy_rot, 'b-', linewidth=2, label=label, alpha=0.7)
        ax.plot(*pred_xy_rot, 'bo', markersize=2)

    # Mark ego position (origin)
    ax.plot(0, 0, 'g*', markersize=15, label='Ego Vehicle', zorder=10)

    ax.set_xlabel('Lateral (meters)', fontsize=10)
    ax.set_ylabel('Forward (meters)', fontsize=10)
    ax.set_title('Predicted Trajectory (Bird\'s Eye View)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Set reasonable view bounds
    all_xy = pred_xy_rot.T
    margin = 10  # meters
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(-margin, all_xy[:, 1].max() + margin)

    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]  # Drop alpha channel

    plt.close(fig)
    return img


def overlay_on_frame(frame, trajectory_img, position='bottom-right', scale=0.3):
    """Overlay trajectory visualization on a video frame."""
    frame = frame.copy()

    # Resize trajectory image
    traj_h, traj_w = trajectory_img.shape[:2]
    new_h = int(traj_h * scale)
    new_w = int(traj_w * scale)

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


def eval_custom_video(
    video_path,
    output_path=None,
    num_inference_frames=4,
    num_output_frames=16,
    num_traj_samples=1,
    motion_type="forward",
    fps=10,
):
    """
    Run Alpamayo-R1 evaluation on a custom video.

    Args:
        video_path: Path to input video file
        output_path: Path for output video (default: auto-generated)
        num_inference_frames: Number of frames to use for inference (default: 4)
        num_output_frames: Number of frames to include in output video (default: 16)
        num_traj_samples: Number of trajectory samples to generate (default: 1)
        motion_type: Type of synthetic egomotion ("stationary", "forward", "curve_left", "curve_right")
        fps: Output video frame rate (default: 10)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = f"eval_{video_path.stem}.mp4"

    print("=" * 70)
    print("ALPAMAYO-R1 CUSTOM VIDEO EVALUATION")
    print("=" * 70)
    print(f"\nInput video: {video_path}")
    print(f"Output video: {output_path}")

    # Load model
    print("\n1. Loading Alpamayo-R1 model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("   Model loaded!")

    # Extract frames for inference
    print("\n2. Extracting frames for inference...")
    inference_frames, video_fps = extract_frames_from_video(
        video_path,
        num_frames=num_inference_frames,
        target_fps=fps
    )

    # Prepare model input
    print("\n3. Preparing model inputs...")

    # Convert frames to torch tensor: (num_frames, H, W, 3) -> (1, num_frames, 3, H, W)
    frames_tensor = torch.from_numpy(inference_frames)
    frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w").unsqueeze(0)

    # Create synthetic egomotion
    ego_history_xyz, ego_history_rot = create_synthetic_egomotion(
        num_history_steps=16,
        motion_type=motion_type
    )

    # Camera index (treat as front wide camera)
    camera_indices = torch.tensor([1], dtype=torch.int64)  # camera_front_wide_120fov

    print(f"  Image frames: {frames_tensor.shape}")
    print(f"  Camera indices: {camera_indices.tolist()}")

    # Create messages and tokenize
    messages = helper.create_message(frames_tensor.flatten(0, 1))
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

    # Run inference
    print("\n4. Running model inference...")
    print(f"   Generating {num_traj_samples} trajectory sample(s)...")

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=256,
            return_extra=True,
        )

    cot = extra["cot"][0]
    print(f"   Chain-of-Causation: {cot}")

    # Save trajectory data
    pred_xyz_np = pred_xyz.cpu().numpy()[0, 0]  # [num_samples, num_steps, 3]

    # Extract more frames for visualization
    print(f"\n5. Extracting {num_output_frames} frames for visualization...")
    output_frames_raw, _ = extract_frames_from_video(
        video_path,
        num_frames=num_output_frames,
        target_fps=fps
    )

    # Create trajectory overlay
    print("\n6. Creating trajectory visualization...")
    traj_img = create_trajectory_plot(pred_xyz_np)

    # Compose frames with overlay
    print("\n7. Compositing frames...")
    output_frames = []

    for i, frame in enumerate(output_frames_raw):
        # Add trajectory overlay
        composed = overlay_on_frame(frame, traj_img, position='bottom-right', scale=0.35)

        # Add text overlays
        composed = add_text_overlay(
            composed,
            f"Frame {i+1}/{num_output_frames}",
            position='top-left'
        )

        # Add reasoning text (truncate if too long)
        reasoning_text = cot[0] if len(cot[0]) < 80 else cot[0][:77] + "..."
        composed = add_text_overlay(
            composed,
            f"Reasoning: {reasoning_text}",
            position='top-center'
        )

        output_frames.append(composed)

    # Save video
    print(f"\n8. Saving video to {output_path}...")

    height, width = output_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in output_frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"   Video saved! ({len(output_frames)} frames @ {fps} fps)")

    # Also save a single comparison image
    comparison_path = output_path.replace('.mp4', '_comparison.png')
    comparison_frame = output_frames[len(output_frames) // 2]
    cv2.imwrite(comparison_path, cv2.cvtColor(comparison_frame, cv2.COLOR_RGB2BGR))
    print(f"   Comparison image saved: {comparison_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print(f"Comparison image: {comparison_path}")
    print(f"Reasoning: {cot[0]}")
    print(f"Predicted trajectory: {pred_xyz_np.shape[1]} waypoints over 6.4 seconds")
    print("\nNote: Predictions use synthetic egomotion data and may be less accurate")
    print("than evaluations on the PhysicalAI-AV dataset with real pose measurements.")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Alpamayo-R1 on custom video files"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output video (default: auto-generated)"
    )
    parser.add_argument(
        "--inference-frames",
        type=int,
        default=4,
        help="Number of frames to use for inference (default: 4)"
    )
    parser.add_argument(
        "--output-frames",
        type=int,
        default=16,
        help="Number of frames in output video (default: 16)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of trajectory samples to generate (default: 1)"
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="forward",
        choices=["stationary", "forward", "curve_left", "curve_right"],
        help="Type of synthetic egomotion (default: forward)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video frame rate (default: 10)"
    )

    args = parser.parse_args()

    try:
        eval_custom_video(
            video_path=args.video_path,
            output_path=args.output,
            num_inference_frames=args.inference_frames,
            num_output_frames=args.output_frames,
            num_traj_samples=args.samples,
            motion_type=args.motion,
            fps=args.fps,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
