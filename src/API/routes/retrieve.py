"""
Receiving pose and keypoint vectors from the client side and retrieving the most
similar ones from the database based on some metric

@author: Angel Villar-Corrales
"""

import os
import base64
import cv2

from http import HTTPStatus
from flask import Blueprint, jsonify, request
from flasgger import swag_from

from schemas.retrieve import RetrieveSchema
# from models.retrive import RetriveModel
from lib.logger import log_function, print_


retrieve_api = Blueprint('api/retrieve', __name__)
@retrieve_api.route('/', methods=['POST'])
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Data processed successfully!',
            'schema': RetrieveSchema
        }
    }
})
@log_function
def receive_data():
    """
    Receives a pose vector and keypoint coordinates to retrieve the similar poses from
    a database
    Receives a pose vector and keypoint coordinates to retrieve the similar poses from
    a database
    ---
    """

    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response, 200
