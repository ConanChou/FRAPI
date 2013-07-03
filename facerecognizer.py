import cv2
from numpy import array
import os

class FaceRecognizer(object):
    """
    Main brain of the FR API.
    """
    def __init__(self, arg):
        super(FaceRecognizer, self).__init__()
        self.arg = arg


if __name__ == '__main__':

    test_dir = 'test_img'
    folders = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]
    folders = filter(lambda x: os.path.isdir(x), folders)

    training_data = []
    responses = []

    cate = 0
    id_dict = {}

    for folder in folders:
        print folder
        id_dict[cate] = folder
        for img_path in os.listdir(folder):
            print "- " + img_path
            img = cv2.imread(os.path.join(folder, img_path))
            training_data.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            responses.append(cate)
        cate += 1

    svm = cv2.SVM(array(training_data), array(responses))

    sample = cv2.imread(os.path.join(test_dir, 'test.jpg'))
    result = svm.predict(sample)

    print result
