"""
Methods for processing the retrieval dabase fit with pose skeletons from all person
instanes in certain dataset, i.e., MS-COCO, Styled-COCO or Vases

@author: Angel Villar-Corrales
"""

import os
import pickle

import numpy as np
import hnswlib

from lib.logger import print_
from CONFIG import CONFIG


def process_pose_vector(vector, approach, normalize=True):
    """
    Processing a pose matrix (17 keypoints, 3 features) into a pose vector to
    perform pose-based retrieval

    Args:
    -----
    vector: numpy array
        Array with shape (17,3) to convert into a pose vector for retrieval
    approach: string
        Approach (keypoints) used to measure similarity
    normalize: boolean
        If True, normalized pose vectors are used
    """

    # obtaining inidices for desired keypoints
    if(approach == "all_kpts"):
        kpt_idx = np.arange(17)  # all keypoints
    elif(approach == "full_body"):
        kpt_idx = np.arange(5, 17)  # from shoulders to hips
    elif(approach == "upper_body"):
        kpt_idx = np.arange(5, 12)  # from shoulders to ankles
    else:
        print(f"ERROR!. Approach {approach}. WTF?")
        exit()

    # removing visibility and sampling only desired keypoints
    processed_vector = vector[kpt_idx, 0:2].flatten()
    dim = processed_vector.shape[-1]
    if(normalize):
        processed_vector /= np.linalg.norm(processed_vector)

    return processed_vector


def load_database(db_name):
    """
    Loading the pickled file with the database data

    Args:
    -----
    db_name: string
        name of the database to load

    Returns:
    --------
    database: dictionary
        dict containing the data from the retrieval database (img_name, annotations, ...)
    """

    db_path = CONFIG["paths"]["database_path"]
    pickle_path = os.path.join(db_path, f"database_{db_name}.pkl")
    with open(pickle_path, "rb") as file:
        database = pickle.load(file)

    return database["data"]


def load_knn(dataset_name, approach, metric="euclidean_distance", normalize=True):
    """
    Loading the prefit knn object for the current retrieval task

    Args:
    -----
    dataset_name: list
        list with the names of the databases used for retrieval
    approach: string
        Approach (keypoints) used to measure similarity
    metric: string
        metric used by the knn object for retrieval
    normalize: boolean
        If True, normalized pose vectors are used

    Returns:
    --------
    knn: HNSW object
        HNSW graph used for the knn retrieval
    data: numpy array
        data samples as fit into the knn object
    """

    # obtaining names and paths of the data and neighbors objects
    knn_dir = CONFIG["paths"]["knn_path"]
    name_mask = f"datasets_{dataset_name}_approach_{approach}_metric_{metric}_"\
                f"norm_{normalize}.pkl"
    knn_path = os.path.join(knn_dir, f"graph_{name_mask}")
    data_path = os.path.join(knn_dir, f"data_{name_mask}")

    # making sure those objects exist
    if(not os.path.exists(knn_path)):
        message = f"KNN path '{knn_path}' does not exists..."
        print_(message, message_type="error")
    if(not os.path.exists(data_path)):
        message = f"KNN data '{data_path}' does not exists..."
        print_(message, message_type="error")

    # loading data
    with open(data_path, "rb") as file:
        data = pickle.load(file)
    knn = hnswlib.Index(space='l2', dim=34)
    knn.load_index(knn_path, max_elements=0)

    return knn, data



#
