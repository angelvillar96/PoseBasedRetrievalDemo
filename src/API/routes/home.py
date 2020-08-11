from http import HTTPStatus
from flask import Blueprint, jsonify
from flasgger import swag_from

from schemas.home import HomeSchema
from models.home import HomeModel

home_api = Blueprint('api', __name__)

@home_api.route('/')
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Retrieval Demo API is running!',
            'schema': HomeSchema
        }
    }
})
def home():
    """
    Check if API is online
    Check if API is online
    ---
    """
    result = HomeModel()
    print(result)
    result = HomeSchema().dump(result)
    print(result)
    response = jsonify(result)

    return response, 200
