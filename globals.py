import cv2
import os
import time
import pickle

def init(model_path, data_path):
    global category
    global is_trained
    global fr
    global before
    category = {}
    is_trained = False
    before = time.time()
    fr = cv2.createLBPHFaceRecognizer()
    if os.path.isfile(model_path) and os.path.isfile(data_path):
        fr.load(model_path)
        with open(data_path, 'rb') as data_backup:
            category = pickle.load(data_backup)


