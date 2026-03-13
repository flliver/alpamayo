#!/usr/bin/env python3
"""
Split a video into segments and evaluate each segment with Alpamayo-R1.

This script:
1. Splits a long video into multiple segments
2. Runs Alpamayo-R1 evaluation on each segment
3. Creates a summary visualization comparing predictions across segments
"""

import argparse
import sys
from pathlib import Path
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from eval_custom_video import eval_custom_video


def get_video_info(video_path):
    """Get video duration and properties."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'width': width,
        'height': height,
    }


def split_video_into_segments(video_path, segment_duration=10, output_dir=None):
    """
    Split video into segments using OpenCV.

    Args:
        video_path: Path to input video
        segment_duration: Duration of each segment in seconds (default: 10)
        output_dir: Directory for segment files (default: same as video)

    Returns:
        List of segment file paths
    """
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_segments"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # Get video info
    info = get_video_info(video_path)
    num_segments = int(np.ceil(info['duration'] / segment_duration))

    print(f"\nSplitting video into {num_segments} segments of {segment_duration}s each...")
    print(f"Video: {info['duration']:.1f}s @ {info['fps']:.1f} FPS")
    print(f"Output directory: {output_dir}")

    segment_paths = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = info['fps']
    width = info['width']
    height = info['height']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min(start_time + segment_duration, info['duration'])
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        output_path = output_dir / f"segment_{i:03d}.mp4"

        print(f"  Segment {i+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s", end='')

        # Create video writer for this segment
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Write frames to segment
        frames_written = 0
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1

        out.release()

        if frames_written > 0:
            print(f" [OK - {frames_written} frames]")
            segment_paths.append(output_path)
        else:
            print(f" [FAILED - no frames]")

    cap.release()

    print(f"\nCreated {len(segment_paths)} segments")
    return segment_paths


def evaluate_segments(
    video_path,
    segment_duration=10,
    num_inference_frames=4,
    num_output_frames=16,
    num_traj_samples=1,
    motion_type="forward",
    fps=10,
):
    """
    Split video into segments and evaluate each segment.

    Args:
        video_path: Path to input video
        segment_duration: Duration of each segment in seconds
        num_inference_frames: Number of frames for model inference
        num_output_frames: Number of frames in output visualization
        num_traj_samples: Number of trajectory samples per segment
        motion_type: Synthetic egomotion type
        fps: Output video frame rate

    Returns:
        List of evaluation results (dicts with segment info and predictions)
    """
    video_path = Path(video_path)

    print("=" * 70)
    print("VIDEO SEGMENTATION AND EVALUATION")
    print("=" * 70)
    print(f"\nInput video: {video_path}")
    print(f"Segment duration: {segment_duration}s")
    print(f"Motion assumption: {motion_type}")

    # Split video into segments
    segment_paths = split_video_into_segments(video_path, segment_duration)

    if not segment_paths:
        print("No segments created!")
        return []

    # Evaluate each segment
    results = []
    output_dir = video_path.parent / f"{video_path.stem}_eval_segments"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print(f"EVALUATING {len(segment_paths)} SEGMENTS")
    print("=" * 70)

    for i, segment_path in enumerate(segment_paths):
        print(f"\n{'='*70}")
        print(f"SEGMENT {i+1}/{len(segment_paths)}: {segment_path.name}")
        print(f"{'='*70}")

        output_path = output_dir / f"eval_segment_{i:03d}.mp4"

        try:
            # Run evaluation
            eval_custom_video(
                video_path=segment_path,
                output_path=str(output_path),
                num_inference_frames=num_inference_frames,
                num_output_frames=num_output_frames,
                num_traj_samples=num_traj_samples,
                motion_type=motion_type,
                fps=fps,
            )

            results.append({
                'segment_id': i,
                'segment_path': str(segment_path),
                'output_path': str(output_path),
                'comparison_path': str(output_path).replace('.mp4', '_comparison.png'),
                'success': True,
            })

        except Exception as e:
            print(f"\n❌ Failed to evaluate segment {i}: {e}")
            results.append({
                'segment_id': i,
                'segment_path': str(segment_path),
                'success': False,
                'error': str(e),
            })

    # Create summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r.get('success', False))
    print(f"\nEvaluated {successful}/{len(results)} segments successfully")

    for i, result in enumerate(results):
        if result.get('success', False):
            print(f"  ✓ Segment {i}: {result['output_path']}")
        else:
            print(f"  ✗ Segment {i}: {result.get('error', 'Unknown error')}")

    # Save results JSON
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split video into segments and evaluate each with Alpamayo-R1"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=10,
        help="Duration of each segment in seconds (default: 10)"
    )
    parser.add_argument(
        "--inference-frames",
        type=int,
        default=4,
        help="Number of frames for model inference (default: 4)"
    )
    parser.add_argument(
        "--output-frames",
        type=int,
        default=16,
        help="Number of frames in output visualization (default: 16)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of trajectory samples per segment (default: 1)"
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="forward",
        choices=["stationary", "forward", "curve_left", "curve_right"],
        help="Synthetic egomotion type (default: forward)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video frame rate (default: 10)"
    )

    args = parser.parse_args()

    try:
        results = evaluate_segments(
            video_path=args.video_path,
            segment_duration=args.segment_duration,
            num_inference_frames=args.inference_frames,
            num_output_frames=args.output_frames,
            num_traj_samples=args.samples,
            motion_type=args.motion,
            fps=args.fps,
        )

        if not any(r.get('success', False) for r in results):
            print("\n❌ All segment evaluations failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
