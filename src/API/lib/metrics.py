"""
Implementation of methods for computing pose similarity metrics for retrieval
"""

import numpy as np

import lib.retrieval_database as retrieval_database


def confidence_score(query, pose_db, confidence):
    """
    Computing the confidence score for pose similarity. This metric weights the distance
    between keypoints with the confidence with which each point was detected

    Args:
    -----
    query, pose_db: numpy array
        pose vectors for the query and database image
    confidence: numpy array
        vector with the confidence  with which each query keypoint was detected
    """

    # normalizing with the sum of confidences so metric is bounded by 1
    confidence = confidence / np.sqrt(np.sum(np.power(confidence,2)))
    norm = 1 / (np.sum(confidence))
    weighted_scores = np.sqrt(np.sum(confidence * np.power(query - pose_db, 2)))
    confidence_score = norm * weighted_scores

    return confidence_score


def oks_score(query, pose_db, approach):
    """
    Computing the object keypoint similarity between two poses. Metric inspired by
    flow-based person tracking in videos

    Args:
    -----
    query, pose_db: numpy array
        pose vectors for the query and database image
    """

    # defining and normalizing variance of the gaussians for each keypojnt
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,
                       .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    sigmas = retrieval_database.process_pose_vector(vector=sigmas, approach=approach,
                                                    normalize=False)

    square_dists = [(query[2*i] - pose_db[2*i])**2 + (query[2*i+1] - pose_db[2*i+1])**2
                    for i in range(len(query) // 2)]
    exponent = square_dists / (np.power(sigmas, 2) * 2)
    oks = np.sum( np.exp(-1 * exponent) ) / (len(query) // 2)

    oks = 1 - oks  # unlike distance, the larger oks the better, so we do this :)

    return oks

#
