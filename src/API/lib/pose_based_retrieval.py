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
from lib.retrieval_database import get_db_img, extract_detection, load_knn,\
    process_pose_vector, get_neighbors_idxs

DB = None           # data and metadata from the retrieval database
FEATURES = None     # processed pose vectors from the database
KNN = None          # 'trained' kNN object used for retrival
KEYS = None         # image names, corresponding to the keys of the DB-dictionary


@log_function
def pose_based_retrieval(kpt_idx, all_keypoints, dataset_name="['coco']", approach="full_body",
                         retrieval_method="euclidean_distance", penalization=None,
                         normalize=True, num_retrievals=10):
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
    keypoints = [keypoints[:, 1], keypoints[:, 0], keypoints[:, 2]]
    keypoints = np.array(keypoints).T
    pose_vector = process_pose_vector(vector=keypoints, approach=approach, normalize=True)
    print(keypoints)

    # loading knn if necessary
    global KNN
    global DB
    global FEATURES
    global KEYS
    if(KNN is None):
        print_("Loading KNN object and retrieval database...")
        KNN, DB, FEATURES = load_knn(dataset_name=dataset_name, approach=approach,
                                     normalize=normalize)
        KEYS = list(DB.keys())

    # retrieving similar poses from database
    print_("Retrieving similar poses from database...")
    idx, dists = get_neighbors_idxs(pose_vector, k=num_retrievals, approach=approach,
                                    retrieval_method=retrieval_method,
                                    penalization=penalization, knn=KNN,
                                    database=FEATURES)
    retrievals = [DB[KEYS[j]] for j in idx]

    # getting data from each of the retrievals
    metadata = {
        "img_name": [],
        "distance": []
    }
    retrieval_paths = []
    for j,ret in enumerate(retrievals):
        savepath = os.path.join(os.getcwd(), "data", "final_results",
                                "retrievals", f"retrieval_{j+1}.png")
        joints, img_name = ret['joints'].numpy(), ret['img']
        center, scale = ret['center'].numpy(), ret['scale'].numpy()
        metadata["img_name"].append(img_name)
        metadata["distance"].append(dists[j].tolist())

        # processing joints for vsualization
        joints[:,-1] = 1  # adding visibility
        joints = [joints[:, 1], joints[:, 0], joints[:, 2]]
        joints = np.array(joints).T

        # laoding image and extracting instance with similar pose
        img = get_db_img(img_name)
        instance = extract_detection(img=img, center=center, scale=scale)

        # saving instance in retrievals directory
        draw_pose(img=instance, poses=[np.arange(17)],
                  all_keypoints=joints, savepath=savepath, savefig=True)
        retrieval_paths.append(savepath)

    return retrieval_paths, metadata
