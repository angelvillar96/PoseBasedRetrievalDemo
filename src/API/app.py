"""
Main app for the Flaks API for the 'Pose-based image retrieval' project

@author: Angel Villar-Corrales
"""

import os
from argparse import ArgumentParser

from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

from routes.home import home_api
from routes.upload import upload_api

def create_app():
    """
    Initialization of the API
    """

    app = Flask(__name__)
    cors = CORS(app)
    app.config['SWAGGER'] = {
        'title': 'Flask API Starter Kit',
    }
    swagger = Swagger(app)

    app.register_blueprint(home_api, url_prefix='/api')
    app.register_blueprint(upload_api, url_prefix='/api/upload')
    app.run(host='0.0.0.0', port=port)
    return


if __name__ == '__main__':
    os.system("clear")
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app = create_app()

#
