import sqlite3
from app import app
from contextlib import closing
from flask import g, url_for
import os, time
import cv2
import cv2.cv as cv
import numpy as np
import sys
import pickle
import globals

def connect_db():
    """return a sqlite3 connection"""
    return sqlite3.connect(app.config['DATABASE'])

def init_db():
    """call it only you want to reset the db"""
    with closing(connect_db()) as db:
        with app.open_resource('FRAPI.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def insert_to_db(type, face_name, orig_img, face_img):
    """insertion query wrapper"""
    cur = g.db.cursor()
    cur.execute('INSERT OR REPLACE INTO images (type, name, orig_img, face_img) values (?, ?, ?, ?)', [type, face_name, orig_img, face_img])
    g.db.commit()
    id = cur.lastrowid
    cur.close()
    return id


def allowed_file(filename):
    """filter for image format"""
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def create_if_not_exists(path):
    """make dirs if not exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_uri(type, face_name, file_name):
    """return full URL of a uploaded image"""
    face_name = '' if face_name is None else face_name
    return app.config['SITE_URL'] + url_for(
        'get_file',
        path=os.path.join(app.config['FR_DIR'][type],
        face_name,
        file_name))


def detect(img, cascade_fn='haarcascades/haarcascade_frontalface_alt.xml',
           scaleFactor=1.3, minNeighbors=4, minSize=(20, 20),
           flags=cv.CV_HAAR_SCALE_IMAGE):
    """detect face and return rectangle axises"""

    cascade = cv2.CascadeClassifier(cascade_fn)
    rects = cascade.detectMultiScale(img, scaleFactor=scaleFactor,
                                     minNeighbors=minNeighbors,
                                     minSize=minSize, flags=flags)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def preprocess(f):
    """extract face from photo"""
    def inner(*args, **kw):
        if len(args) < 4:
            face_name = ''
            type, file, file_name = args
        else:
            type, file, file_name, face_name = args
        img_color = cv2.imread(file)
        img_gray = cv2.cvtColor(img_color, cv.CV_RGB2GRAY)
        img_gray = cv2.equalizeHist(img_gray)

        rects = detect(img_gray)
        crop_img(img_color, rects, face_name, file_name, type)
        return f(*args, **kw)
    return inner

def crop_img(img, rects, face_name, file_name, type):
    """crop a image from a given rectangle axises"""
    create_if_not_exists(os.path.join(app.config['FR_DIR'][type], face_name))
    for x1, y1, x2, y2 in rects:
        cv2.imwrite(os.path.join(app.config['FR_DIR'][type], face_name, file_name), img[y1:y2, x1:x2])


def backup_model(f):
    """backup decorator"""
    def inner(*args, **kw):
        f(*args, **kw)
        now = time.time()
        if now - globals.before > app.config['BACKUP_INTERVAL']:
            globals.fr.save(app.config['MODEL_BACKUP_NAME'])
            with open(app.config['APP_DATA_BACKUP_NAME'], 'wb') as data_backup:
                pickle.dump(globals.category, data_backup)
            globals.before = now
    return inner

@backup_model
@preprocess
def train_model(type, upload_save_dir, file_name, face_name):
    """train face recognition model"""
    imgs, cats = read_image(face_name, file_name, 'training')
    if globals.is_trained:
        globals.fr.update(np.asarray(imgs), np.asarray(cats))
    else:
        globals.fr.train(np.asarray(imgs), np.asarray(cats))
        globals.is_trained = True

@preprocess
def predict(type, upload_save_dir, file_name):
    """get prediction from a trained face recognition model"""
    imgs, cats = read_image(None, file_name, 'testing')
    [p_label, p_confidence] = globals.fr.predict(np.asarray(imgs[0]))
    if p_confidence < app.config['CONFIDENCE_THRESHOLD'] and p_confidence != 0:
        return {'message': 'face not recognized'}
    names = [name for name, category in globals.category.items() if category == p_label]
    return {'prediction': names[0] if len(names) > 0 else 'N/A',
            'confidence': p_confidence if p_confidence != 0 else 100}

def read_image(face_name, file_name, type, size=(100,100)):
    """read image accordingly"""
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

