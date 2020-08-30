import cv2
import numpy as np
import tensorflow as tf
import numpy.random as npr

from settings import *
from utils import get_iou
from mtcnn.detect import Detector

DETECTORS = {}

def gen_neg_img(img, boxes, nums=50):
    """
    产生negative图片
    :param img: 原图
    :param boxes: 人脸框， shape=(-1, 4)
    :param nums: 总需产生个数
    :return: tuple， 裁剪后的图片和标签
    """
    h, w, _ = img.shape

    have = 0
    while nums:
        # 随机裁剪
        size = npr.randint(12, max(min(h, w)//2, 13))
        x = npr.randint(0, w-size)
        y = npr.randint(0, h-size)

        iou = get_iou(np.array([x, y, size, size]), boxes)
        if np.max(iou) < 0.3:
            yield img[y:y+size, x:x+size], 0, [0.] * 4
            have += 1

        # # 在boxes周围裁剪
        # size = npr.randint(boxes[:, 2:]*0.7-1, boxes[:, 2:]*1.3)
        # size = np.maximum([12, 12], size)

        # point = npr.randint(boxes[:, :2]-size*0.3-1, boxes[:, :2]+size*0.3)
        # point = np.maximum([0, 0], point)

        # crop_box = np.hstack([point, size])

        # for box in crop_box:
        #     if box[0] + box[2] >= w or box[1] + box[3] >= h:
        #         continue
        #     iou = get_iou(box, boxes)
        #     if np.max(iou) >= 0.3:
        #         continue
        #     x, y, ww, hh = box
        #     yield img[y:y+hh, x:x+ww], 0, [0.] * 4
        #     have += 1
        
        nums -= 1

    return have


def gen_crop(img, boxes, landmark=None, display=False):

    height, width, _ = img.shape

    # 每一种类别的数量， pos, part, neg, landmark
    crop_nums = [0] * 4

    # 先随机产生一定数量neg-img
    if landmark is None:
        crop_nums[2] += yield from gen_neg_img(img, boxes)        

    # 在每个人脸框附近产生每个类别的数据
    for box in boxes:
        x1, y1, w, h = box

        # 忽略小脸
        if min(w, h) < 20 or x1 < 0 or y1 < 0:
            continue

        if landmark is not None:
            yield img[y1: y1+h, x1: x1+w], -2, (*[0.] * 4, (landmark - box[:2]) / box[3:]), 

        # 每个框判断
        for i in range(15):

            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.2 * max(w, h)))

            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)


            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))

            nx2 = nx1 + size
            ny2 = ny1 + size
            if nx2 > width or ny2 > height:
                continue

            crop_box = np.array([nx1, ny1, size, size])

            # crop
            cropped_im = img[ny1: ny2, nx1: nx2]

            box_ = box.reshape(1, -1)
            iou = get_iou(crop_box, box_)

            if iou < 0.4:
                continue
            
            # yu gt de offset
            offset_x1 = (x1 - nx1) / size
            offset_y1 = (y1 - ny1) / size

            offset_x2 = (x1+w-nx1-size) / size
            offset_y2 = (y1+h-ny1-size) / size

            if iou >= 0.65:
                if landmark is None:
                    yield cropped_im, 1, (offset_x1, offset_y1, offset_x2, offset_y2)
                    crop_box[0] += 1
                else:
                    marks = (landmark - crop_box[:2]) / crop_box[3:]
                    yield cropped_im, -2, (offset_x1, offset_y1, offset_x2, offset_y2, marks)
                    crop_box[3] += 1

            elif landmark is None and iou >= 0.4:
                yield cropped_im, -1, (offset_x1, offset_y1, offset_x2, offset_y2)
                crop_box[1] += 1

    if display:
        print("pos: %d part: %d neg: %d lanmark: %d" % tuple(crop_box))


def gen_hard(net, img, boxes):
    if net not in DETECTORS:
        DETECTORS[net] = Detector(0.5, 0.5, net=net)
    detector = DETECTORS[net]
    bbox, _ = detector.predict(img, 20, net=net)
    crop_nums = [0] * 3
    for box in bbox:
        crop_img = img[box[1]: box[3], box[0]: box[2]]

        box[2] -= box[0]
        box[3] -= box[1]

        iou = get_iou(box, boxes)

        max_ind = np.argmax(iou)

        if iou[max_ind] < 0.3:
            if crop_nums[0] < 50:
                yield crop_img, 0, [0.] * 4
                crop_nums[0] += 1
            continue
        if iou[max_ind] < 0.4:
            continue

        true_box = boxes[max_ind]
        x1, y1, w, h = true_box
        
        offset_x1 = (x1 - box[0]) / float(box[2])
        offset_y1 = (y1 - box[1]) / float(box[3])

        offset_x2 = (x1+w-box[0]-box[2]) / float(box[2])
        offset_y2 = (y1+h-box[1]-box[3]) / float(box[3])

        if iou[max_ind] >= 0.65:
            yield crop_img, 1, (offset_x1, offset_y1, offset_x2, offset_y2)
            crop_nums[1] += 1

        else:
            yield crop_img, -1, (offset_x1, offset_y1, offset_x2, offset_y2)
            crop_nums[2] += 1

    # print(crop_nums)



if __name__ == '__main__':
    pass

