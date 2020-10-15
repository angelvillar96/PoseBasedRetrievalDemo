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
from lib.metrics import confidence_score, oks_score
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
        kpt_idx = np.append(kpt_idx, 0)
    elif(approach == "upper_body"):
        kpt_idx = np.arange(5, 12)  # from shoulders to ankles
        kpt_idx = np.append(kpt_idx, 0)
    else:
        print(f"ERROR!. Approach {approach}. WTF?")
        exit()

    # removing visibility and sampling only desired keypoints
    if(len(vector.shape) > 1):
        processed_vector = vector[kpt_idx, 0:2].flatten()
    else:
        processed_vector = vector[kpt_idx]
    dim = processed_vector.shape[-1]
    if(normalize):
        norm = np.linalg.norm(processed_vector)
        epsilon = 1e-5
        norm = norm if norm > epsilon else 1e-5
        processed_vector = processed_vector / norm

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
    elif(img_name[0] in ["w", "e", "v", "b"]):
        data_dir = "arch_data"
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
    features_path = os.path.join(knn_dir, f"features_{name_mask}")

    # making sure those objects exist
    if(not os.path.exists(knn_path)):
        message = f"KNN path '{knn_path}' does not exists..."
        print_(message, message_type="error")
        exit()
    if(not os.path.exists(data_path)):
        message = f"KNN data '{data_path}' does not exists..."
        print_(message, message_type="error")
        exit()
    if(not os.path.exists(features_path)):
        message = f"KNN features '{features_path}' does not exists..."
        print_(message, message_type="error")
        exit()

    # determining dimensionality of the feature vectors
    if(approach == "full_body"):
        dim = 26
    elif(approach == "all_kpts"):
        dim = 34
    if(approach == "upper_body"):
        dim = 14

    # loading data
    with open(data_path, "rb") as file:
        data = pickle.load(file)
    with open(features_path, "rb") as file:
        features = pickle.load(file)
    knn = hnswlib.Index(space='l2', dim=dim)
    knn.load_index(knn_path, max_elements=0)

    return knn, data, features


def get_neighbors_idxs(query, num_retrievals=10, approach="full_body",
                       retrieval_method="knn", penalization=None, **kwargs):
    """
    Iterating the database measuring distance from query to each dataset element
    and retrieving the elements with the smallest distance. Not Optimized

    Args:
    -----
    query: numpy array
        pose vector used as retrieval query
    k: integer
        number of elements to retrieve from the dataset
    approach: string
        strategy followed for the retrieval procedure
    penalization: string
        strategy followed to penalize the non-detected keypoints

    Returns:
    --------
    idx: list
        indices of the retrieved elements
    dist: list
        distances from the query to the database elements
    """

    # kNN retrieval approach. Just giving the query to the fit kNN graph => O(log(N))
    if(retrieval_method == "knn"):
        if("knn" in kwargs):
            print_("ERROR! 'knn' object was not given as parameter")
            exit()
        knn = kwargs["knn"]
        idx, dists = knn.knn_query(query, k=num_retrievals)
        idx, dists = idx[0,:], dists[0,:]
        return idx, dists

    # defining method for computing metrics other than knn
    elif(retrieval_method == "euclidean_distance"):
        compute_metric = lambda x,y,z: np.sqrt(np.sum(np.power(x - y, 2)))
        confidence = np.ones(query.shape)
    elif(retrieval_method == "manhattan_distance"):
        compute_metric = lambda x,y,z: np.abs(np.sum(x - y))
        confidence = np.ones(query.shape)
    elif(retrieval_method == "confidence_score"):
        if("scores" not in kwargs):
            print_("ERROR! Parameters 'scores' must be given to use to 'confidence_score' " \
                   "as a retrieval metric...")
            exit()
            # confidence = np.ones(query.shape)
        else:
            confidence = kwargs["scores"]
        compute_metric = lambda x,y,z: confidence_score(x, y, z)
    elif(retrieval_method == "oks_score"):
        # sigmas for gaussian kernel evaluated at each keypoint
        confidence = np.ones(query.shape)
        compute_metric = lambda x,y,z: oks_score(x, y, approach)
    else:
        print_(f"ERROR! Retrieval metric '{retrieval_method}' is not defined...")
        exit()

    # other methods require iterating the dataset => O(N)
    assert "database" in kwargs, "ERROR! 'database' object was not given as parameter"
    database = kwargs["database"]
    n_vectors, dims = database.shape

    if(penalization in ["mean", "max"]):
        penalization_value = get_penalization_metric(query=query, database=database,
                                                     penalization=penalization,
                                                     metric_func=compute_metric,
                                                     confidence=confidence)
    epsilon = 1e-5
    dists = []
    # print(query)
    for i, pose_vect in enumerate(database):

        # applying penalizations to ocluded points if necessary
        # ocluded points are assigned coordinate (0,0)
        if(penalization == "zero_coord"):
            cur_query = query
            cur_confidence = confidence
            cur_vect = pose_vect
        # removing kpts that are ocluded either in query or database item
        elif(penalization == "none" or penalization is None):
            cur_query, cur_confidence = np.copy(query), np.copy(confidence)
            cur_vect = np.copy(pose_vect)
            # obtainign idx of kpts that are 0 in query
            idx = np.where(np.abs(query) < epsilon)[0]
            cur_query[idx], cur_vect[idx], cur_confidence[idx] = 0, 0, 0
        # assigning mean/max value of metrics to ocluded points
        elif(penalization in ["mean", "max"]):
            cur_query, cur_confidence = np.copy(query), np.copy(confidence)
            cur_vect = np.copy(pose_vect)
            # obtainign idx of kpts that are 0 in (query AND NOT(db))
            idx = np.where((np.abs(query) < epsilon) & (np.abs(cur_vect) > epsilon))[0]
            cur_query[idx] = penalization_value
            cur_vect[idx], cur_confidence[idx] = 0, 0
        # computing specified metric and saving 'distance' results
        dist = compute_metric(cur_query, cur_vect, cur_confidence)
        dists.append(dist)

    idx = np.argsort(dists)[:num_retrievals]
    dists = [dists[i] for i in idx]

    return idx, dists


def get_penalization_metric(query, database, metric_func, penalization="mean",
                            confidence=None, N=100):
    """
    Computing the mean or max distance between query and database

    Args:
    -----
    query, database: np arrays
        pose vectors corresponding to the query and the database
    metric_func: function
        function used to compute the metric between vectors
    penalization: string
        type of penalization to apply: ['mean', 'max']
    confidence: numpy array
        vector with the confidence  with which each query keypoint was detected
    N: integer
        number of database elements considered to compute penalization_value
    """

    assert penalization in ["mean", "max"]

    dists = []
    for i, cur_vect in enumerate(database):
        if(i==N):
            break
        dist = metric_func(query, cur_vect, confidence)
        dists.append(dist)

    if(penalization == "mean"):
        return np.mean(dists)
    elif(penalization == "max"):
        return np.max(dists)



#
