import torch
import cv2
import qai_hub as hub
from qai_hub_models.models.mediapipe_pose import MediaPipePoseDetector,MediaPipePoseLandmarkDetector

# Load the model
pose_detector_model = MediaPipePoseDetector.from_pretrained()
pose_landmark_detector_model = MediaPipePoseLandmarkDetector.from_pretrained()

# Connect to cloud-hosted device (e.g., Samsung Galaxy S23)
device = hub.Device("Samsung Galaxy S23")




def extract_frames(video_path, frame_rate=5):
    """Extract frames from a video at a given frame rate."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_rate == 0:
            cv2.imshow('frame',frame)
            frames.append(frame)
        count += 1

    cap.release()
    return frames


def detect_pose(frames, model):
    """Detect poses in each video frame."""
    results = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        output = model(rgb_frame)
        results.append(output)
    return results



def draw_keypoints(frame, keypoints):
    """Overlay keypoints on a video frame."""
    for kp in keypoints:
        x, y, score = kp['x'], kp['y'], kp['score']
        if score > 0.5:
            cv2.circle(
                frame, 
                (int(x * frame.shape[1]), int(y * frame.shape[0])), 
                5, (0, 255, 0), -1
            )
    return frame


def process_video(video_path, output_path, frame_rate=5):
    frames = extract_frames(video_path, frame_rate)
    pose_results = detect_pose(frames, pose_detector_model)

    # Set up video writer
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height)
    )

    # Process each frame and write to output video
    for frame, result in zip(frames, pose_results):
        keypoints = result[0]['keypoints']
        frame_with_keypoints = draw_keypoints(frame, keypoints)
        out.write(frame_with_keypoints)

    out.release()
    print(f"Processed video saved at {output_path}")


input_video = "yoga_video.mp4"  # Replace with your video path
output_video = "output_yoga_pose.mp4"
process_video(input_video, output_video)

