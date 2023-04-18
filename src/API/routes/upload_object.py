"""
Receiving the data from the client side and processing the image

@author: Angel Villar-Corrales
"""

import os
import base64
import cv2

from http import HTTPStatus
from flask import Blueprint, jsonify, request
from flasgger import swag_from

from schemas.upload import UploadSchema
# from models.upload import UploadModel
from lib.utils import timestamp, save_image_from_post, encode_img, create_directory
from lib.logger import log_function, print_
from lib.object_detection import setup_detector, object_detection


upload_object_api = Blueprint('api/upload_object', __name__)
@upload_object_api.route('/', methods=['POST'])
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Data processed successfully!',
            'schema': UploadSchema
        }
    }
})
@log_function
def receive_data():
    """
    Receives the image and metadata from a JSON and processing it
    Receives the image and metadata from a JSON and processing it
    ---
    """

    # relevant variables extracted from the POST data
    print_("Route '/api/upload_object' was called...")
    data = request.form
    files = request.files
    file_name = data["file_name"]
    object_detector = data["object_detector"]
    database = data["retrieval_database"]
    # print(list(data.keys()), database, data["database"])

    # relevant paths/dirs for storing results
    imgs_path = os.path.join(os.getcwd(), "data", "imgs")
    create_directory(imgs_path)
    full_dets_path = os.path.join(os.getcwd(), "data", "final_results_objects", "full_dets")
    create_directory(full_dets_path)
    file_path = os.path.join(imgs_path, file_name)
    final_path = os.path.join(full_dets_path, file_name)

    # saving data in directory, cause why not?
    print_("Storing query in data directory...")
    storage_element = files["file"]
    save_image_from_post(data=storage_element, path=file_path)

    # object detection
    det_img_path, det_instances_path,\
        det_data, det_labels = object_detection(img_path=file_path,
                                    object_detector=object_detector,
                                    database=database)

    # saving final results
    img = cv2.imread(det_img_path, cv2.IMREAD_COLOR)
    cv2.imwrite(final_path, img)

    # sending the data back to the user. Encoding images to binary format and arrays
    # as lists so as to form a json response
    print_("Encoding results and returning response...")
    encoded_img = encode_img(path=final_path)
    encoded_dets = [encode_img(path=det_path) for det_path in det_instances_path]

    # print(encoded_dets)
    json_data = {
        "img_name": os.path.basename(file_path),
        "img_url": file_path,
        "img_binary": encoded_img,
        "detections": encoded_dets,
        "labels": det_labels
    }
    response = jsonify(json_data)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response, 200


#
