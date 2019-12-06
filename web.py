#!/usr/bin/env python3

""" Basic flask web app to return generated images """
from flask import Flask, request, send_from_directory
import os
import sys
import torch
from ganutils import visualize

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    """ This will return a single png image."""
    content = request.get_json()
    name = content['name']
    path = os.path.join('results', name)
    if not os.path.exists(path):
        # todo- Need to do an actual 404
        return '404 - session not found'



    return content['name']


@app.route('/images/<img>')
def serve_image(img):
    return send_from_directory('images',img)

if __name__ == '__main__':
    app.run()
