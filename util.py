import sqlite3
from app import app
from contextlib import closing
from flask import g, url_for
import os
import cv2
import cv2.cv as cv
import numpy as np
import sys
import globals

def connect_db():
    return sqlite3.connect(app.config['DATABASE'])

def init_db():
    with closing(connect_db()) as db:
        with app.open_resource('FRAPI.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def insert_to_db(type, face_name, orig_img, face_img):
    cur = g.db.cursor()
    cur.execute('INSERT OR REPLACE INTO images (type, name, orig_img, face_img) values (?, ?, ?, ?)', [type, face_name, orig_img, face_img])
    g.db.commit()
    id = cur.lastrowid
    cur.close()
    return id

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_uri(type, face_name, file_name):
    face_name = '' if face_name is None else face_name
    return app.config['SITE_URL'] + url_for(
        'get_file',
        path=os.path.join(app.config['FR_DIR'][type],
        face_name,
        file_name))

def detect(img, cascade_fn='haarcascades/haarcascade_frontalface_alt.xml',
           scaleFactor=1.3, minNeighbors=4, minSize=(20, 20),
           flags=cv.CV_HAAR_SCALE_IMAGE):

    cascade = cv2.CascadeClassifier(cascade_fn)
    rects = cascade.detectMultiScale(img, scaleFactor=scaleFactor,
                                     minNeighbors=minNeighbors,
                                     minSize=minSize, flags=flags)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def preprocess(type, file, file_name, face_name=''):
    img_color = cv2.imread(file)
    img_gray = cv2.cvtColor(img_color, cv.CV_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)

    rects = detect(img_gray)
    crop_img(img_color, rects, face_name, file_name, type)

def crop_img(img, rects, face_name, file_name, type):
    create_if_not_exists(os.path.join(app.config['FR_DIR'][type], face_name))
    for x1, y1, x2, y2 in rects:
        cv2.imwrite(os.path.join(app.config['FR_DIR'][type], face_name, file_name), img[y1:y2, x1:x2])

def train_model(face_name, file_name):
    imgs, cats = read_image(face_name, file_name, 'training')
    if globals.is_trained:
        globals.fr.update(np.asarray(imgs), np.asarray(cats))
    else:
        globals.fr.train(np.asarray(imgs), np.asarray(cats))
        globals.is_trained = True

def predict(file_name):
    imgs, cats = read_image(None, file_name, 'testing')
    return globals.fr.predict(np.asarray(imgs[0]))

def read_image(face_name, file_name, type, size=(100,100)):
    imgs, cats = [], []

    if face_name:
        length = len(globals.category)
        if face_name not in globals.category:
            cats.append(length)
            globals.category[face_name] = length
        else:
            cats.append(globals.category[face_name])
        cats = np.asarray(cats, dtype=np.int32)
    else:
        face_name = ''
    try:
        im = cv2.imread(os.path.join(app.config['FR_DIR'][type], face_name, file_name),
                cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, size)
        imgs.append(np.asarray(im, dtype=np.uint8))
    except IOError, (errno, strerror):
        print "I/O error({0}): {1}".format(errno, strerror)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise
    return imgs, cats

