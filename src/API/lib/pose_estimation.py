"""
Methods for pose estimation and keypoint detection

@author: Angel Villar-Corrales
"""

import os

import numpy as np
import cv2
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision.transforms as transforms

from lib.logger import log_function, print_
from lib.pose_parsing import get_final_preds_hrnet, get_max_preds_hrnet, create_pose_entries
from lib.visualizations import draw_pose

KEYPOINT_ESTIMATOR = None
NORMALIZER = None

@log_function
def setup_pose_estimator():
    """
    Initializing the pretrained pose estimation model
    """

    print_("Initializing Pose Estimation model...")
    model = models.PoseHighResolutionNet(is_train=False)

    print_("Loading pretrained model parameters...")
    pretrained_path = os.path.join(os.getcwd(), "resources", "coco_hrnet_w32_256x192.pth")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.eval()

    # intiializing preprocessing method
    global NORMALIZER
    NORMALIZER = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

    return model


@log_function
def pose_estimation(detections, centers, scales):
    """
    Detecting the keypoints for the input images and parsing the human poses

    Args:
    -----
    detections: numpy array
        person dectections from the input images. Shape is (n_dets, 3, 256, 192)
    centers: numpy array
        center coordinates of each person detection
    scales: numpy array
        scale factor for each person detection

    Returns:
    --------
    TODO
    """

    # initializing the model if necessary
    global KEYPOINT_ESTIMATOR
    if(KEYPOINT_ESTIMATOR is None):
        KEYPOINT_ESTIMATOR = setup_pose_estimator()

    # preprocessing the detections
    print_("Preprocessing person detections...")
    norm_detections = [NORMALIZER(torch.Tensor(det)).numpy() for det in detections]

    # forward pass through the keypoint detector model
    print_("Computing forward pass through the keypoint detector model...")
    keypoint_dets = KEYPOINT_ESTIMATOR(torch.Tensor(norm_detections).float())
    scaled_dets = F.interpolate(keypoint_dets.clone(), (256, 192),
                                mode="bilinear", align_corners=True)

    # extracting keypoint coordinates and confidence values from heatmaps
    print_("Extracting keypoints from heatmaps...")
    keypoint_coords, max_vals_coords = get_max_preds_hrnet(scaled_heats=scaled_dets.numpy())
    keypoints, max_vals, coords = get_final_preds_hrnet(heatmaps=keypoint_dets.numpy(),
                                                        center=centers, scale=scales)

    # parsing poses by combining and joining keypoits
    print_("Parsing human poses...")
    indep_pose_entries, indep_all_keypoints = create_pose_entries(keypoints=keypoint_coords,
                                                                  max_vals=max_vals_coords,
                                                                  thr=0.1)
    pose_entries, all_keypoints = create_pose_entries(keypoints=keypoints,
                                                      max_vals=max_vals,
                                                      thr=0.1)

    # creating pose visualizations and saving in the corresponding directory
    # TODO

    return


#
