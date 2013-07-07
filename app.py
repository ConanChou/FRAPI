from flask import Flask, jsonify, request, g, url_for, make_response
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from werkzeug import secure_filename
from util import *
import os
import globals

globals.init()

__all__ = ['make_json_app']

def make_json_app(import_name, **kwargs):
    """
    Creates a JSON-oriented Flask app.

    All error responses that you don't specifically
    manage yourself will have application/json content
    type, and will contain JSON like this (just an example):

    { "message": "405: Method Not Allowed" }
    """
    def make_json_error(ex):
        response = jsonify(message=str(ex))
        response.status_code = (ex.code
                                if isinstance(ex, HTTPException)
                                else 500)
        return response

    app = Flask(import_name, **kwargs)

    for code in default_exceptions.iterkeys():
        app.error_handler_spec[None][code] = make_json_error

    return app

app = make_json_app(__name__)
app.config.from_object('config')


@app.before_request
def before_request():
    g.db = connect_db()

@app.teardown_request
def teardown_request(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()

@app.route('/face_recognizer')
def hello_world():
    return 'Hello World!'

@app.route('/face_recognizer/api/v1/faces/training', methods=['POST'])
def add_training_data():
    type = 'training'
    face_name = secure_filename(request.values['name'])
    create_if_not_exists(os.path.join(app.config['UPLOAD_DIR'][type], face_name))
    file = request.files['file']
    if face_name and file and allowed_file(file.filename):
        file_name = secure_filename(file.filename)

        upload_save_dir = os.path.join(app.config['UPLOAD_DIR'][type], face_name, file_name)
        training_save_dir = os.path.join(app.config['FR_DIR'][type], face_name, file_name)

        file.save(upload_save_dir)
        preprocess(type, upload_save_dir, file_name, face_name)

        id = insert_to_db(type, face_name, upload_save_dir, training_save_dir)

        train_model(face_name, file_name)

        return jsonify({
            'id'    : id,
            'uri'   : get_uri(type, face_name, file_name),
            'name'  : face_name
            }), 201
    return 400

@app.route('/face_recognizer/api/v1/faces/testing', methods=['POST'])
def add_testing_data():
    type = 'testing'
    file = request.files['file']
    if file and allowed_file(file.filename):
        file_name = secure_filename(file.filename)

        upload_save_dir = os.path.join(app.config['UPLOAD_DIR'][type], file_name)
        testing_save_dir = os.path.join(app.config['FR_DIR'][type], file_name)

        file.save(upload_save_dir)
        preprocess(type, upload_save_dir, file_name)

        id = insert_to_db(type, None, upload_save_dir, testing_save_dir)

        [p_label, p_confidence] = predict(file_name)

        if p_confidence < 70 and p_confidence != 0:
            return jsonify({
                'id'     : id,
                'uri'    : get_uri(type, None, file_name),
                'message': 'face not recognized'
                }), 201

        names = [name for name, category in globals.category.items() if category == p_label]

        return jsonify({
            'id'        : id,
            'uri'       : get_uri(type, None, file_name),
            'prediction': names[0],
            'confidence': p_confidence
            }), 201
    return 400


@app.route('/face_recognizer/api/v1/faces/<int:face_id>')
def get_face(face_id):
    pass

@app.route('/images/<path:path>')
def get_file(path):
    resp = make_response(open('uploads/' + path).read())
    resp.content_type = 'image/jpeg'
    return resp


if __name__ == '__main__':
    app.run()
