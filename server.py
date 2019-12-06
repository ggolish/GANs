#!/usr/bin/env python3

import json

from flask import Flask, request
from flask_cors import CORS

from ganutils import trainer

app = Flask(__name__, static_folder='frontend')
CORS(app)

results = {}


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/get-sessions', methods=['POST'])
def get_sessions():
    sessions = trainer.get_sessions()
    response = json.dumps({'sessions': sessions})
    return response


@app.route('/load-session', methods=['POST'])
def load_session():
    global results
    response = {}
    if request.method == 'POST':
        try:
            data = request.get_json()
            if data['name'] not in results:
                results[data['name']] = trainer.load_results(data['name'])
                print(results)
            response['status'] = 'success'
        except Exception as e:
            response['status'] = repr(e)
    else:
        response['status'] = f'invalid method {request.method}'

    return json.dumps(response)


if __name__ == '__main__':
    app.run()
