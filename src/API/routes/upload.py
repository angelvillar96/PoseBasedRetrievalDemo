"""
Receiving the data from the client side and processing the image

@author: Angel Villar-Corrales
"""

from http import HTTPStatus
from flask import Blueprint, jsonify, request
from flasgger import swag_from

from schemas.upload import UploadSchema
# from models.upload import UploadModel

upload_api = Blueprint('api/upload', __name__)
@upload_api.route('/', methods=['POST'])
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Data processed successfully!',
            'schema': UploadSchema
        }
    }
})
def receive_data():
    """
    Receives the image and metadata from a JSON and processing it
    Receives the image and metadata from a JSON and processing it
    ---
    """

    data = request.form
    files = request.files

    return response, 200


#
