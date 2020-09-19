import os
import sys

import tensorflow as tf
import numpy as np
import cv2

from mtcnn.model import PNet, RNet, ONet
from settings import CHECKPOINT_DIR
from utils import mark_img, nms, convert_to_square, pad, rect_img


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Detector:
    def __init__(self, p_pro=0.5, r_pro=0.6, o_pro=0.6, load=True, net=3):
        assert 0 < p_pro < 1 and 0 < r_pro < 1 and 0 < o_pro < 1, '传的什么玩意'
        self._p_pro, self._r_pro, self._o_pro = p_pro, r_pro, o_pro
        self.pnet = PNet()
        self.rnet = self.onet = None

        if net >= 2:
            self.rnet = RNet()
        if net >= 3:
            self.onet = ONet()
        
        if load:
            self.load_weights()

    def load_weights(self):
        try:
            self.pnet.load_weights(os.path.join(CHECKPOINT_DIR, 'p_net', 'cp.ckpt'))
            self.rnet.load_weights(os.path.join(CHECKPOINT_DIR, 'r_net', 'cp.ckpt'))
            self.onet.load_weights(os.path.join(CHECKPOINT_DIR, 'o_net', 'cp.ckpt'))
        except Exception as e:
            print(e)

    def _pnet_detect(self, inputs, minsize=20, scale_factor=0.709):
        bboxes = []
        scores = []

        # 以12*12为1个单元，将最小人脸调整成12*12大小，而后图像金字塔检测缩放至12，即从检测多个人脸到最后检测一个人脸
        img = self._img_resize(inputs, 12/minsize)

        # 图像金字得到所有预选框
        while min(img.shape[:2]) >= 12:
            cls, reg = self.pnet.predict(tf.expand_dims(img, 0))

            bbox, score = self._get_box(reg, cls[0, :, :, 1], img.shape[0]/inputs.shape[0])     

            img = self._img_resize(img, scale_factor)

            if len(bbox) == 0:
                continue

            keep = nms(bbox, score, 0.5, 'union')

            bboxes.append(bbox[keep])
            scores.append(score[keep])

        if not bboxes:
            return np.empty((0, 4)), np.empty((0, 1))

        bboxes = np.vstack(bboxes)
        scores = np.hstack(scores)

        # 将金字塔后的图片再进行一次抑制, 此时主要避免重合
        keep = nms(bboxes, scores, 0.7, 'min')
        bboxes = bboxes[keep]
        scores = scores[keep]

        return bboxes, scores

    def _rnet_detect(self, inputs, boxes):
        """
        通过PNet结果修正人脸框后送入RNet网络
        :param boxes:
        :return:
        """
        # 将pnet的box变成包含它的正方形，可以避免信息损失
        convert_to_square(boxes)

        rnet_box = Detector._pad(inputs, boxes, 24)

        cls, reg = self.rnet.predict(rnet_box)

        keep = np.where(cls[:, 1] > self._r_pro)
        if keep[0].size == 0:
            return np.empty((0, 4)), np.empty((0, 1))

        scores = cls[:, 1][keep]

        boxes = Detector._get_reg_box(boxes[keep], reg[keep])
        keep = nms(boxes, scores, 0.7, 'union')
        return boxes[keep], scores[keep]

    def _onet_detect(self, inputs, boxes):
        convert_to_square(boxes)

        onet_box = Detector._pad(inputs, boxes, 48)

        cls, reg, marks = self.onet.predict(onet_box)

        keep = np.where(cls[:, 1] > self._o_pro)
        if keep[0].size == 0:
            return np.empty((0, 4)), np.empty((0, 1)), np.empty((0, 10))

        scores = cls[:, 1][keep]
        marks = marks[keep]
        boxes = boxes[keep]

        marks = np.reshape(marks, (-1, 5, 2))
        marks[:, :, 0] = marks[:, :, 0] * (boxes[:, 2:3] - boxes[:, 0:1]) + boxes[:, 0:1]
        marks[:, :, 1] = marks[:, :, 1] * (boxes[:, 3:4] - boxes[:, 1:2]) + boxes[:, 1:2]
        marks = np.int32(marks)

        boxes = Detector._get_reg_box(boxes, reg[keep])
        keep = nms(boxes, scores, 0.7, 'min')
        return boxes[keep], scores[keep], marks[keep]

    def predict(self, inputs: np.ndarray, minsize=50, scale_factor=0.709, net=3):
        inputs = (inputs - 127.5) / 128
        bbox, score = self._pnet_detect(inputs, minsize, scale_factor)

        if net == 1 or len(bbox) == 0:
            return bbox, score

        bbox, score = self._rnet_detect(inputs, bbox)

        if net == 2 or len(bbox) == 0:
            return bbox, score

        bbox, score, marks = self._onet_detect(inputs, bbox)

        return bbox, score, marks

    def _get_box(self, bbox, score, scale, cellsize=12, stride=2):
        idx = np.where(score > self._p_pro)
        score = score[idx]
        bbox = np.vstack([np.int32((stride * idx[1]) / scale),
                          np.int32((stride * idx[0]) / scale),
                          np.int32((stride * idx[1] + cellsize) / scale),
                          np.int32((stride * idx[0] + cellsize) / scale),
                          ])
        return bbox.T, score

    @staticmethod
    def _get_reg_box(box, reg):
        # box的长宽
        bbw = box[:, 2] - box[:, 0] + 1
        bbh = box[:, 3] - box[:, 1] + 1
        # 对应原图的box坐标
        boxes_c = np.vstack([box[:, 0] + reg[:, 0] * bbw,
                             box[:, 1] + reg[:, 1] * bbh,
                             box[:, 2] + reg[:, 2] * bbw,
                             box[:, 3] + reg[:, 3] * bbh]).T
        boxes_c = np.maximum(0, boxes_c.astype(np.int32))
        return boxes_c

    @staticmethod
    def _img_resize(img, scale):
        h, w, _ = img.shape

        shape = (int(w*scale), int(h*scale))

        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _pad(im, bboxes, size):
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(bboxes, im.shape[0], im.shape[1])
        num_boxes = bboxes.shape[0]
        cropped_ims = np.zeros((num_boxes, size, size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3))
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = cv2.resize(tmp, (size, size), interpolation=cv2.INTER_AREA)

        return cropped_ims


if __name__ == '__main__':
    det = Detector()
    for img_name in sys.argv[1:]:
        img = cv2.imread(img_name)
        boxes, score, *marks = det.predict(img, net=3)

        print(len(boxes), 'faces has been detected!!')
        img = rect_img(img, boxes)

        # 关键点位置
        if marks:
            img = mark_img(img, marks[0])

        cv2.imwrite(img_name.split('.')[0]+'_ed.png', img)




