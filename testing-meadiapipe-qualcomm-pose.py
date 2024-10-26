import torch

import qai_hub as hub
from qai_hub_models.models.mediapipe_pose import MediaPipePoseDetector,MediaPipePoseLandmarkDetector

# Load the model
pose_detector_model = MediaPipePoseDetector.from_pretrained()
pose_landmark_detector_model = MediaPipePoseLandmarkDetector.from_pretrained()

# Device
device = hub.Device("Samsung Galaxy S23")

# Trace model
pose_detector_input_shape = pose_detector_model.get_input_spec()
pose_detector_sample_inputs = pose_detector_model.sample_inputs()

traced_pose_detector_model = torch.jit.trace(pose_detector_model, [torch.tensor(data[0]) for _, data in pose_detector_sample_inputs.items()])

# Compile model on a specific device
pose_detector_compile_job = hub.submit_compile_job(
    model=traced_pose_detector_model ,
    device=device,
    input_specs=pose_detector_model.get_input_spec(),
)

# Get target model to run on-device
pose_detector_target_model = pose_detector_compile_job.get_target_model()
# Trace model
pose_landmark_detector_input_shape = pose_landmark_detector_model.get_input_spec()
pose_landmark_detector_sample_inputs = pose_landmark_detector_model.sample_inputs()

traced_pose_landmark_detector_model = torch.jit.trace(pose_landmark_detector_model, [torch.tensor(data[0]) for _, data in pose_landmark_detector_sample_inputs.items()])

# Compile model on a specific device
pose_landmark_detector_compile_job = hub.submit_compile_job(
    model=traced_pose_landmark_detector_model ,
    device=device,
    input_specs=pose_landmark_detector_model.get_input_spec(),
)

# Get target model to run on-device
pose_landmark_detector_target_model = pose_landmark_detector_compile_job.get_target_model()
