# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import cv2
import numpy as np
from pathlib import Path
import os
import time

from app import MediaPipePoseApp
from qai_hub_models.models.mediapipe_pose.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipePose,
)
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

POSE_LANDMARK_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 13),
    (13, 15),
    (15, 17),
    (17, 19),
    (19, 15),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 16),
    (16, 22),
    (11, 12),
    (12, 24),
    (24, 23),
    (23, 11),
]

def process_video(app: MediaPipePoseApp, input_path: str, output_path: str, frames_dir: str) -> None:
    """
    Process a video file and extract one frame every second with pose detection.
    
    Args:
        app: MediaPipePoseApp instance
        input_path: Path to input video file
        output_path: Path to save processed video
        frames_dir: Directory to save individual frames
    """
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video Info:")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    
    # Calculate frames to skip (1 second worth of frames)
    frames_to_skip = fps
    
    # Create frames directory if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_count = 0
    time_point = 0  # Time point in seconds
    start_time = time.time()
    
    while cap.isOpened():
        # Set frame position to the exact second we want
        cap.set(cv2.CAP_PROP_POS_FRAMES, time_point * fps)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with pose detection
        processed_frame = app.predict_landmarks_from_image(frame_rgb)[0]
        
        # Convert back to BGR for saving
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{time_point:04d}sec.jpg")
        cv2.imwrite(frame_path, processed_frame_bgr)
        
        # Update progress
        print(f"Extracted frame at {time_point} seconds ({(time_point/duration)*100:.1f}% complete)")
        
        # Move to next second
        time_point += 1
        if time_point * fps >= total_frames:
            break
    
    # Release resources
    cap.release()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Total frames extracted: {time_point}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing speed: {time_point/processing_time:.2f} frames per second")
    print(f"Extracted frames saved to: {frames_dir}")

def main(is_test: bool = False, video_path: str = ''):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default=video_path,
        help="Path to input video file",
    )
    add_output_dir_arg(parser)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.75,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )

    args = parser.parse_args([] if is_test else None)

    print("Starting video processing...")
    
    # Load app
    app = MediaPipePoseApp(
        MediaPipePose.from_pretrained(), args.score_threshold, args.iou_threshold,
    )
    print("Model loaded successfully")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create frames directory within output directory
    frames_dir = output_dir / "frames_1sec"
    
    # Generate output video path
    input_path = Path(args.video)
    output_path = output_dir / f"{input_path.stem}_processed{input_path.suffix}"
    
    # Process video
    process_video(app, str(input_path), str(output_path), str(frames_dir))

if __name__ == "__main__":
    main(video_path='/Users/apple/Pictures/qidk-megathon/videoplayback.mp4')