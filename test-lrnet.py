import cv2
import numpy as np
from qai_hub_models.models.litehrnet.app import LiteHRNetApp
from qai_hub_models.models.litehrnet.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    LiteHRNet,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
import time
from PIL import Image
from qai_hub_models.utils.draw import draw_points

IA_HELP_MSG = "More inferencer architectures for litehrnet can be found at https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/topdown_heatmap/coco"

def draw_skeleton(frame, keypoints):
    """
    Draw skeleton on the frame using the 17 COCO keypoints
    
    Args:
        frame: Original frame (numpy array)
        keypoints: Array of shape (17, 2) containing keypoint coordinates
        
    Keypoint indices:
    0: nose
    1: left_eye
    2: right_eye
    3: left_ear
    4: right_ear
    5: left_shoulder
    6: right_shoulder
    7: left_elbow
    8: right_elbow
    9: left_wrist
    10: right_wrist
    11: left_hip
    12: right_hip
    13: left_knee
    14: right_knee
    15: left_ankle
    16: right_ankle
    """
    output_frame = frame.copy()
    
    # Define the connections between keypoints for skeleton
    skeleton = [
        # Face
        (0, 1), (0, 2), (1, 3), (2, 4),  # Nose to eyes, eyes to ears
        # Upper body
        (5, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12),  # Shoulders to hips
        # Lower body
        (11, 12),  # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16)  # Right leg
    ]
    
    # Colors for different parts
    colors = {
        'face': (255, 200, 0),  # Cyan
        'upper_body': (255, 255, 0),  # Yellow
        'lower_body': (0, 255, 255)  # Yellow
    }
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        # Different colors for different body parts
        if i <= 4:  # Face keypoints
            color = colors['face']
        elif i <= 10:  # Upper body
            color = colors['upper_body']
        else:  # Lower body
            color = colors['lower_body']
            
        cv2.circle(output_frame, (int(x), int(y)), 4, color, -1)
        # Add keypoint labels for better visualization
        cv2.putText(output_frame, str(i), (int(x), int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw connections
    for pair in skeleton:
        pt1 = tuple(map(int, keypoints[pair[0]]))
        pt2 = tuple(map(int, keypoints[pair[1]]))
        
        # Choose color based on body part
        if pair[0] <= 4 and pair[1] <= 4:
            color = colors['face']
        elif pair[0] <= 10 or pair[1] <= 10:
            color = colors['upper_body']
        else:
            color = colors['lower_body']
            
        cv2.line(output_frame, pt1, pt2, color, 2)
    
    return output_frame

def process_video(video_source, app, output_path=None, display=True):
    """
    Process video stream with LiteHRNet pose estimation.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source {video_source}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB for model input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process frame
            start_time = time.time()
            
            # Get keypoints with raw_output=True
            keypoints = app.predict_pose_keypoints(pil_image, raw_output=True)[0]
            
            
            # # Draw skeleton on frame
            # output_frame = draw_skeleton(frame, keypoints)
            draw_points(frame, keypoints,color=(0, 255, 0),size=15)
            
            
            # Calculate and display FPS
            fps_current = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f'FPS: {fps_current:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame if saving video
            if writer:
                writer.write(frame)
            
            # Display frame
            if display:
                cv2.imshow('LiteHRNet Pose Estimation', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(LiteHRNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--video_source",
        type=str,
        default="0",
        help="video file path or camera index (default: 0 for webcam)",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="save output video",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="disable real-time display",
    )
    
    args = parser.parse_args([] if is_test else None)
    litehrnet_model = model_from_cli_args(LiteHRNet, args)
    hub_model = demo_model_from_cli_args(LiteHRNet, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Initialize model and app
    app = LiteHRNetApp(hub_model, litehrnet_model.inferencer)
    print("Model Loaded")

    # Process video
    video_source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    output_path = None
    if args.save_video:
        output_path = f"{args.output_dir}/output_video.mp4" if args.output_dir else "output_video.mp4"
    
    process_video(
        video_source=video_source,
        app=app,
        output_path=output_path,
        display=not args.no_display
    )

if __name__ == "__main__":
    main()