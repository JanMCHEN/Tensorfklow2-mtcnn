import cv2
import time

from mtcnn.detect import Detector
from utils import rect_img, mark_img


def main(fps=0, show=False):
    cap = cv2.VideoCapture(0)
    detect = Detector()
    if not fps:
        fps = 50
    delay = 1000//fps
    while True:
        st = time.time()
        ret, img = cap.read()

        if not ret:
            break

        boxes, score, *marks = detect.predict(img, net=3)
        if show:
            print(f'{len(boxes)} faces,detect time:', time.time()-st)

        img = rect_img(img, boxes)
        if marks:
            img = mark_img(img, marks[0])

        cv2.imshow('video', img)
        q = cv2.waitKey(delay)

        if q == 27:
            print('exit')
            break

if __name__ == '__main__':
    main(30)
