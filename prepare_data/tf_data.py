'''
Author: your name
Date: 2020-08-11 16:17:08
LastEditTime: 2020-08-20 09:25:48
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \MTCNN-TensorFlow-2-master\data\tf_data.py
'''
import sys
import tensorflow as tf
import cv2
import numpy as np
import numpy.random as npr

from prepare_data.pre_crop_img import gen_crop, gen_hard
from settings import *
from utils import gen_face_img_data, get_iou, gen_lanmark_img_data

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


class DateSet:
    def __init__(self, size=12):
        self.size = size

        self._writers = None 
        self._dataset = None

    def __init_writers(self, mark=False):
        if self._writers is not None:
            return
        
        if not mark:
            self._writers = {
                -1: tf.io.TFRecordWriter(PART_RECORDS[self.size]),
                0: tf.io.TFRecordWriter(NEG_RECORDS[self.size]),
                1: tf.io.TFRecordWriter(POS_RECORDS[self.size]),
            }
        else:
            self._writers = {-2: tf.io.TFRecordWriter(MARK_RECORD)}

    def __init_dataset(self):
        if self._dataset is not None:
            return
        self._dataset = {
            'mark': tf.data.TFRecordDataset(MARK_RECORD).map(self._decode_face(True)).repeat(),
            'pos': tf.data.TFRecordDataset(POS_RECORDS[self.size]).map(self._decode_face()).repeat(),
            'part': tf.data.TFRecordDataset(PART_RECORDS[self.size]).map(self._decode_face()).repeat(),
            'neg': tf.data.TFRecordDataset(NEG_RECORDS[self.size]).map(self._decode_face()).repeat(),
        }


    def get_writer(self, clss):
        return self._writers[clss]


    def _tf_save(self, img, clss, box, marks=None):
        feature = {
            'img':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.image.encode_png(img).numpy()])),
            'clss': tf.train.Feature(int64_list=tf.train.Int64List(value=[clss])),
            'box':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(box, tf.float32)).numpy()]))
        }
        if marks is not None:
            feature['marks'] =  tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(marks, tf.float32)).numpy()]))

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
        
        writer = self.get_writer(clss)
        writer.write(example_proto)
            

    def save_face_box(self):
        self.__init_writers()
        for img, boxes in tqdm(gen_face_img_data(FACE_ANN_FILE, FACE_IMG_DIR)):
            for cropped, clss, box in gen_crop(img, boxes):
                cropped = cv2.resize(cropped, (self.size, self.size), interpolation=cv2.INTER_AREA)
                self._tf_save(cropped, clss, box)

    
    def save_mark_face(self):
        self.__init_writers(mark=True)
        for img, boxes, marks in tqdm(gen_lanmark_img_data(MARK_ANN_FILE, MARK_IMG_DIR)):
            for cropped, clss, (*box, marks) in gen_crop(img, boxes, marks):
                cropped = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_AREA)
                self._tf_save(cropped, clss, box, marks)
        
    def save_hard_face(self):
        self.__init_writers()
        net = self.size // 48 + 1
        for img, boxes in tqdm(gen_face_img_data(FACE_ANN_FILE, FACE_IMG_DIR)):
            for cropped, clss, box in gen_hard(net, img, boxes):
                cropped = cv2.resize(cropped, (self.size, self.size), interpolation=cv2.INTER_AREA)
                self._tf_save(cropped, clss, box)

    def _decode_face(self, mark=False):
        # example_proto = tf.train.Example.FromString(example_proto)
        _feature_description = {
            'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'clss': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'box': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'marks': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        def map(example_proto):
            example_dict = tf.io.parse_single_example(example_proto, _feature_description)

            img = tf.image.decode_png(example_dict['img'])
            if not mark:
                marks = [0.] * 10
                img = img = tf.cast(img, tf.float32)
            else:
                img = tf.image.resize(img, (self.size, self.size), 'area')
                marks = tf.io.parse_tensor(example_dict['marks'], tf.float32)
                marks = tf.reshape(marks, (10, ))
            
            img = (img - 127.5) / 128.

            clss = example_dict['clss']
            box = tf.io.parse_tensor(example_dict['box'], tf.float32)       
                
            return img, clss, box, marks

        return map
    

    def _merge(self, *dataset):
        imgs = tf.concat([ds[0] for ds in dataset], axis=0)
        # 随机亮度
        imgs = tf.image.random_brightness(imgs, 0.3)
        # 随机对比度
        imgs = tf.image.random_contrast(imgs, 0.7, 1.5)

        # 随机饱和度
        imgs = tf.image.random_saturation(imgs, 0.5, 2)
        
        clss = tf.concat([ds[1] for ds in dataset], axis=0)
        box = tf.concat([ds[2] for ds in dataset], axis=0)
        marks = tf.concat([ds[3] for ds in dataset], axis=0)
        return imgs, clss, box, marks


    def get_dataset(self, batch_size=10):
        # pos:part:neg:landmark=1:1:3:2
        self.__init_dataset()
        pos = self._dataset['pos'].shuffle(batch_size*10).batch(batch_size)
        neg = self._dataset['neg'].shuffle(batch_size*10).batch(batch_size*3)
        part = self._dataset['part'].shuffle(batch_size*10).batch(batch_size)
        mark = self._dataset['mark'].shuffle(batch_size*10).batch(batch_size*2)
        return tf.data.Dataset.zip((pos, neg, part, mark)).map(self._merge).prefetch(2)


if __name__ == '__main__':
    size = 12
    if len(sys.argv) > 0:
        size = int(sys.argv[1])

    dataset = DateSet(size)

    if sys.argv[-1] == 'face':
        if size == 12:
            dataset.save_face_box()
        else:
            dataset.save_hard_face()
    elif sys.argv[-1] == 'mark':
        dataset.save_mark_face()
    
    # dataset.save_mark_face()
    # dataset.save_face_box()
    # dataset.save_hard_face()
    
    # for i in dataset.dataset_for_pnet(1):
    #     print(i)

    # for i in dataset.get_datasets()['neg'].take(1):
    #     print(i)

    # for i in dataset.dateset_for_other().take(10):
    #     print(i[1])

