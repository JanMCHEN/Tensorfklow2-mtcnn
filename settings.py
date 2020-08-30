'''
Author: your name
Date: 2020-08-11 16:17:08
LastEditTime: 2020-08-19 18:07:29
LastEditors: Please set LastEditors
Description: 配置文件
FilePath: \MTCNN-TensorFlow-2-master\settings.py
'''

import os


# 项目目录
BASE_DIR = os.path.dirname(__file__)


# 数据集目录
DATA_DIR = os.path.join(BASE_DIR, 'data', "WIDER_train")

# 人脸框数据集标注文件
FACE_ANN_FILE = os.path.join(DATA_DIR, 'wider_face_split', 'wider_face_train_bbx_gt.txt')

# 人脸框数据集图片目录
FACE_IMG_DIR = os.path.join(DATA_DIR, 'images')

# 人脸特征点数据集图片目录
MARK_IMG_DIR = DATA_DIR

# 人脸特征点数据集标注文件
MARK_ANN_FILE = os.path.join(DATA_DIR, 'trainImageList.txt')

# tfrecord
POS_RECORDS = {size: os.path.join(BASE_DIR, 'data', 'tfrecord', str(size), 'pos.tfrecord') for size in [12, 24, 48]}
NEG_RECORDS = {size: os.path.join(BASE_DIR, 'data', 'tfrecord', str(size), 'neg.tfrecord') for size in [12, 24, 48]}
PART_RECORDS = {size: os.path.join(BASE_DIR, 'data', 'tfrecord', str(size), 'part.tfrecord') for size in [12, 24, 48]}
MARK_RECORD = os.path.join(BASE_DIR, 'data', 'tfrecord', 'mark.tfrecord') # 48

# 模型权重文件目录
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'data', 'checkpoints')

# 新建一些文件夹，防止路径错误
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
for i in POS_RECORDS:
    os.makedirs(os.path.dirname(POS_RECORDS[i]), exist_ok=True)


