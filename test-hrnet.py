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
from typing import List

from qai_hub_models.models.hrnet_pose.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    HRNetPose,
)
from hrnet_skelton import HRNetPoseAppWithSkeleton
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

def get_media_files(input_dir: str, media_type: str) -> List[Path]:
    """
    Get all video or image files from the input directory.
    
    Args:
        input_dir: Directory path to search for files
        media_type: Either 'video' or 'image'
    
    Returns:
        List of Path objects for the media files
    """
    input_path = Path(input_dir)
    if media_type == 'video':
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
    else:  # image
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    files = []
    for ext in extensions:
        files.extend(input_path.glob(f'*{ext}'))
        files.extend(input_path.glob(f'*{ext.upper()}'))
    
    return sorted(files)

def process_video(app: HRNetPoseAppWithSkeleton, video_path: Path, output_dir: Path, draw_skeleton: bool = True) -> None:
    """
    Process a video file and extract one frame per second with pose detection.
    
    Args:
        app: HRNetPoseAppWithSkeleton instance
        video_path: Path to input video file
        output_dir: Directory to save processed frames
        draw_skeleton: Whether to draw skeleton lines between keypoints
    """
    print(f"\nProcessing video: {video_path.name}")
    
    # Create output directory for this video
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video Info:")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    
    start_time = time.time()
    time_point = 0  # Time point in seconds
    
    while cap.isOpened():
        # Set frame position to the exact second we want
        cap.set(cv2.CAP_PROP_POS_FRAMES, time_point * fps)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process frame with pose detection
        result = app.predict_pose_keypoints(
            frame,
            draw_skeleton=draw_skeleton,
            point_color=(255, 0, 0),  # Red points
            skeleton_color=(0, 255, 0),  # Green skeleton
            point_size=8,
            line_thickness=2
        )[0]
        
        # Convert result to numpy array if needed
        if not isinstance(result, np.ndarray):
            frame_with_pose = np.array(result)
        else:
            frame_with_pose = result
            
        # Convert RGB to BGR if needed
        if frame_with_pose.shape[-1] == 3:
            frame_with_pose = cv2.cvtColor(frame_with_pose, cv2.COLOR_RGB2BGR)
        
        # Save frame
        frame_path = video_output_dir / f"frame_{time_point:04d}sec.jpg"
        cv2.imwrite(str(frame_path), frame_with_pose)
        
        # Update progress
        print(f"Extracted frame at {time_point} seconds ({(time_point/duration)*100:.1f}% complete)")
        
        # Move to next second
        time_point += 1
        if time_point * fps >= total_frames:
            break
    
    # Release resources
    cap.release()
    
    processing_time = time.time() - start_time
    print(f"\nVideo processing complete:")
    print(f"- Total frames extracted: {time_point}")
    print(f"- Processing time: {processing_time:.2f} seconds")
    print(f"- Processing speed: {time_point/processing_time:.2f} frames per second")
    print(f"- Frames saved to: {video_output_dir}")

def process_image(app: HRNetPoseAppWithSkeleton, image_path: Path, output_dir: Path, draw_skeleton: bool = True) -> None:
    """
    Process a single image with pose detection.
    
    Args:
        app: HRNetPoseAppWithSkeleton instance
        image_path: Path to input image file
        output_dir: Directory to save processed image
        draw_skeleton: Whether to draw skeleton lines between keypoints
    """
    print(f"\nProcessing image: {image_path.name}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image file: {image_path}")
        return
    
    # Process image with pose detection
    result = app.predict_pose_keypoints(
        image,
        draw_skeleton=draw_skeleton,
        point_color=(255, 0, 0),  # Red points
        skeleton_color=(0, 255, 0),  # Green skeleton
        point_size=8,
        line_thickness=2
    )[0]
    
    # Convert result to numpy array if needed
    if not isinstance(result, np.ndarray):
        image_with_pose = np.array(result)
    else:
        image_with_pose = result
        
    # Convert RGB to BGR if needed
    if image_with_pose.shape[-1] == 3:
        image_with_pose = cv2.cvtColor(image_with_pose, cv2.COLOR_RGB2BGR)
    
    # Save processed image
    output_path = output_dir / f"pose_{image_path.name}"
    cv2.imwrite(str(output_path), image_with_pose)
    print(f"Processed image saved to: {output_path}")

def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(HRNetPose)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to input directory containing videos or images",
    )
    parser.add_argument(
        "--media-type",
        type=str,
        choices=['video', 'image'],
        required=True,
        help="Type of media to process: 'video' or 'image'",
    )
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="disable skeleton visualization",
    )

    args = parser.parse_args([] if is_test else None)
    
    # Set on_device attribute if missing
    if not hasattr(args, 'on_device'):
        args.on_device = False
    
    validate_on_device_demo_args(args, MODEL_ID)

    print("Starting media processing...")
    
    # Load model
    model = demo_model_from_cli_args(HRNetPose, MODEL_ID, args)
    app = HRNetPoseAppWithSkeleton(model)
    print("Model loaded successfully")

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "pose_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of media files
    media_files = get_media_files(args.input_dir, args.media_type)
    
    if not media_files:
        print(f"No {args.media_type} files found in {args.input_dir}")
        return
    
    print(f"\nFound {len(media_files)} {args.media_type} files to process")
    
    # Process each file
    for file_path in media_files:
        if args.media_type == 'video':
            process_video(app, file_path, output_dir, not args.no_skeleton)
        else:
            process_image(app, file_path, output_dir, not args.no_skeleton)
    
    print("\nAll processing complete!")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()