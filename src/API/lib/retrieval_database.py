"""
Methods for processing the retrieval dabase fit with pose skeletons from all person
instanes in certain dataset, i.e., MS-COCO, Styled-COCO or Vases

@author: Angel Villar-Corrales
"""

import os
import pickle

import numpy as np
import cv2
import hnswlib

from lib.logger import print_
from lib.transforms import get_affine_transform


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
    if(normalize):
        processed_vector = processed_vector / np.linalg.norm(processed_vector)

    return processed_vector


def get_db_img(img_name):
    """
    Loading a full image from the database that has been considered similar by the
    retrieval algorithm

    Args:
    -----
    img_name: string
        name of the image to load

    Returns:
    --------
    img: numpy array
        complete database image with the instance whose pose was considered as similar
    """

    # determining the database from the image name
    if("stylized" in img_name):
        data_dir = "styled_val2017"
    else:
        data_dir = "val2017"
    img_path = os.path.join(os.getcwd(), "database", "imgs", data_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def extract_detection(img, center, scale, shape=(192,256)):
    """
    Extracting the person with the similar pose to the query from the complete image

    Args:
    -----
    img: numpy array
        complete database image with the instance whose pose was considered as similar
    center: numpy array
        center coordinates of the person instance
    scale: numpy array
        scale factor of the person instance to match desired shape
    shape: tuple
        size of the crop for the person instace

    Returns:
    --------
    instance: numpy array
        array with the person instance whose pose was considered as similar to the query
    """


    # obtaining cv2 transform to obtain crop from coordinates
    trans = get_affine_transform(center=center, scale=scale,
                                 rot=0, output_size=shape)
    instance = cv2.warpAffine(img, trans, shape, flags=cv2.INTER_LINEAR)

    return instance



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
    knn_dir =os.path.join(os.getcwd(), "database", "knns")
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
