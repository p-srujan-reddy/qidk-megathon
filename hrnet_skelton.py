# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from mmpose.apis import MMPoseInferencer
from mmpose.codecs.utils import refine_keypoints
from PIL.Image import Image, fromarray

from qai_hub_models.utils.draw import draw_points, draw_connections
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.models.hrnet_pose.app import HRNetPoseApp , get_max_preds

# Define COCO keypoint connections for skeleton visualization
COCO_SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12),  # limbs
    (11, 12),  # hips
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 6),  # shoulders
    (5, 11), (6, 12),  # torso
    (1, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # face and neck
]

class HRNetPoseAppWithSkeleton(HRNetPoseApp):
    """
    Extended version of HRNetPoseApp that adds skeleton visualization.
    """

    def predict_pose_keypoints(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | List[Image],
        raw_output=False,
        draw_skeleton=True,
        point_color=(255, 0, 0),  # Red for points
        skeleton_color=(0, 255, 0),  # Green for skeleton
        point_size=6,
        line_thickness=2
    ) -> np.ndarray | List[Image]:
        """
        Predicts pose keypoints and optionally draws skeleton for a person in the image.

        Parameters:
            pixel_values_or_image: Input image in various formats
            raw_output: If True, returns keypoint coordinates instead of visualized image
            draw_skeleton: If True, draws skeleton lines connecting keypoints
            point_color: Color for keypoints in RGB format
            skeleton_color: Color for skeleton lines in RGB format
            point_size: Size of keypoint circles
            line_thickness: Thickness of skeleton lines

        Returns:
            If raw_output is true:
                keypoints: np.ndarray, shape [B, N, 2]
            Otherwise:
                predicted_images: List[PIL.Image] with drawn keypoints and skeleton
        """
        (NHWC_int_numpy_frames, proc_inputs, x) = self.preprocess_input(
            pixel_values_or_image
        )

        # Run inference
        heatmaps = self.model(x)
        heatmaps = heatmaps.detach().numpy()

        # Create predictions from heatmap
        pred_kps, scores = get_max_preds(heatmaps)

        # Get bbox information
        bbox = proc_inputs["data_samples"][0].gt_instances.bboxes[0]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        scale = proc_inputs["data_samples"][0].gt_instances.bbox_scales[0]

        # Refine keypoints
        keypoints = refine_keypoints(pred_kps, np.squeeze(heatmaps))

        # Scale keypoints to input resolution
        scale_factor = np.array([4.0, 4.0])
        keypoints = keypoints * scale_factor

        # Map coordinates back to original image
        input_size = proc_inputs["data_samples"][0].metainfo["input_size"]
        keypoints = keypoints / input_size * scale + center - 0.5 * scale
        keypoints = np.round(keypoints).astype(np.int32)

        if raw_output:
            return keypoints

        predicted_images = []
        for i, img in enumerate(NHWC_int_numpy_frames):
            # Draw skeleton first (so points appear on top)
            if draw_skeleton:
                # Convert RGB colors to BGR for OpenCV
                bgr_skeleton_color = skeleton_color[::-1]
                draw_connections(
                    img, 
                    keypoints[i], 
                    COCO_SKELETON_CONNECTIONS,
                    color=bgr_skeleton_color,
                    size=line_thickness
                )
            
            # Draw points
            bgr_point_color = point_color[::-1]
            draw_points(
                img, 
                keypoints[i], 
                color=bgr_point_color,
                size=point_size
            )
            
            predicted_images.append(fromarray(img))
        
        return predicted_images