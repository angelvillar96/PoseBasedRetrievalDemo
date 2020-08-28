"""
Logic for pose-based image retrieval: loading the retrieval model/method, processing
query-image to extract pose vector, fetching most similar poses from database based
on some similarity metric.

@author: Angel Villar-Corrales
"""

import os

import numpy as np
import cv2
import hnswlib

from lib.logger import log_function, print_
from lib.visualizations import draw_pose
from lib.retrieval_database import load_knn, process_pose_vector

DB = None
KNN = None
KEYS = NONE

@log_function
def pose_based_retrieval(kpt_idx, all_keypoints, dataset_name="coco", approach="all_kpts",
                         metric="euclidean_distance", normalize=True, num_retrievals=10):
    """
    Main orquestrator for the pose based retrieval functionality

    Args:
    -----
    kpt_idx: list
        list with the idx of the keypoints corresponding to the selected pose
    all_keypoints: numpy array
        array (n_kpts, 4) containing the features of all keypoints detected in the image
    """

    # obtaining only keypoints corresponding to the selected pose and preprocessing
    print_("Preprocessing pose vector...")
    kpt_idx = np.array(kpt_idx)
    keypoints = all_keypoints[kpt_idx,:-1]
    pose_vector = process_pose_vector(vector=keypoints, approach=approach, normalize=True)

    # loading knn if necessary
    global KNN
    global DB
    global KEYS
    if(KNN is None):
        print_("Loading KNN object and retrieval database...")
        KNN, DB = load_knn(dataset_name=dataset_name, approach=approach,
                           metric=metric, normalize=normalize)
        KEYS = list(DB.keys())

    # retrieving similar poses from database
    print_("Retrieving similar poses from database...")
    idx, dists = knn.knn_query(pose_vector, k=num_retrievals)
    idx, dists = idx[0,:], dists[0,:]
    retrievals = [DB[keys_list[j]] for j in idx]

    # getting data from each of the retrievals
    metadata = {
        "img_name": [],
        "distance": []
    }
    for j,ret in enumerate(retrievals):
        savepath = os.path.join(os.getcwd(), "data", "final_results",
                                "retrievals", f"retrieval_{j+1}.png")
        joints, img_name = ret['joints'], ret['img']
        center, scale = ret['center'], ret['scale']
        metadata["img_name"].append(img_name)
        metadata["distance"].append(dists[j])


    return
