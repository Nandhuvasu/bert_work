import os
import time
import re
import pathlib
from uuid import uuid4
from env_vars import constants
import falcon
from falcon_swagger_ui import register_swaggerui_app
import bert.bert_vectorizer as bv

# begin by downloading models
bv.download_and_load_model()
# falcon.API instances are callable WSGI apps
APP = falcon.API()

# Setup swagger
SWAGGERUI_URL = '/swagger'
SCHEMA_URL = 'static/swagger.json'
STATIC_PATH = pathlib.Path(__file__).parent / 'static'
# APP.add_static_route('/static', str(STATIC_PATH))
APP.add_static_route('/static', os.path.abspath('static'))
register_swaggerui_app(
    APP, SWAGGERUI_URL, SCHEMA_URL
)
# register_swaggerui_app(APP, '/api-docs', 'static/swagger.json')

# route for BERT
APP.add_route('/v1/vectorizer', bv.BertVectorizer())
