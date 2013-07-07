import cv2

def init():
    global category
    global is_trained
    global fr
    category = {}
    is_trained = False
    fr = cv2.createLBPHFaceRecognizer()
