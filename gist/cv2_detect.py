import cv2
import cv2.cv as cv


def detect(img, cascade_fn='haarcascades/haarcascade_frontalface_alt.xml',
           scaleFactor=1.3, minNeighbors=4, minSize=(20, 20),
           flags=cv.CV_HAAR_SCALE_IMAGE):

    cascade = cv2.CascadeClassifier(cascade_fn)
    rects = cascade.detectMultiScale(img, scaleFactor=scaleFactor,
                                     minNeighbors=minNeighbors,
                                     minSize=minSize, flags=flags)
    print rects
    #cv2.imwrite("tmp/1.jpg", img[rects[0][0]:rects[0][1], rects[0][2]:rects[0][3]])
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects
