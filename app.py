from flask import Flask, jsonify, request, g, url_for, make_response, render_template, redirect
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from werkzeug import secure_filename
from utils import *
import os
import globals


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

globals.init(app.config['MODEL_BACKUP_NAME'], app.config['APP_DATA_BACKUP_NAME'])

@app.before_request
def before_request():
    g.db = connect_db()

@app.teardown_request
def teardown_request(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    return redirect(url_for('tester_index', page='trainer'))

@app.route('/face_recognizer/')
def tester_default():
    return redirect(url_for('tester_index', page='trainer'))

@app.route('/face_recognizer/<page>')
def tester_index(page):
    if page == 'trainer':
        return render_template('training.html', page_name=page, site_url=app.config['SITE_URL'])
    else:
        return render_template('testing.html', page_name=page)

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

        id = insert_to_db(type, face_name, upload_save_dir, training_save_dir)

        train_model(type, upload_save_dir, file_name, face_name)

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
    create_if_not_exists(os.path.join(app.config['UPLOAD_DIR'][type]))
    if file and allowed_file(file.filename):
        file_name = secure_filename(file.filename)

        upload_save_dir = os.path.join(app.config['UPLOAD_DIR'][type], file_name)
        testing_save_dir = os.path.join(app.config['FR_DIR'][type], file_name)

        file.save(upload_save_dir)

        id = insert_to_db(type, None, upload_save_dir, testing_save_dir)
        return_msg = {'id': id, 'uri': get_uri(type, None, file_name)}
        return_msg.update(predict(type, upload_save_dir, file_name))
        return jsonify(return_msg), 201
    return 400

@app.route('/images/<path:path>')
def get_file(path):
    resp = make_response(open('uploads/' + path).read())
    resp.content_type = 'image/jpeg'
    return resp


if __name__ == '__main__':
    app.run()
