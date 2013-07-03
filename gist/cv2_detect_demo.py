import time
from cv2_detect import detect
import cv2
import cv2.cv as cv


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def crop_rects(img, rects):
    for x1, y1, x2, y2 in rects:
        w = x2 - x1
        h = y2 - y1
        crop_img = img[x1:y1, w:h]
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(0)

def demo(in_fn, out_fn):
    print ">>> Loading image..."
    img_color = cv2.imread(in_fn)
    img_gray = cv2.cvtColor(img_color, cv.CV_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    print in_fn, img_gray.shape

    print ">>> Detecting faces..."
    start = time.time()
    rects = detect(img_gray)
    end = time.time()
    print 'time:', end - start
    img_out = img_color.copy()
    i = 0
    for x1, y1, x2, y2 in rects:
        cv2.imwrite("tmp/%d.jpg" % i, img_out[y1:y2, x1:x2])
        i += 1
    draw_rects(img_out, rects, (0, 255, 0))
    cv2.imwrite(out_fn, img_out)


def main():
    demo('../test.jpg', 'pic.detect.jpg')


if __name__ == '__main__':
    main()
