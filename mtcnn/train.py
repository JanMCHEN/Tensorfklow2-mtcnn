'''
Author: your name
Date: 2020-08-11 16:17:08
LastEditTime: 2020-08-21 09:02:34
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \MTCNN-TensorFlow-2-master\mtcnn\train.py
'''
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from mtcnn.model import ONet, PNet, RNet
from settings import *
from prepare_data.tf_data import DateSet

MODEL = {12: PNet, 24: RNet, 48: ONet}


class Train:
    def __init__(self, size, learning_rate=4e-4, checkpoint_path=None, load=False):
        self.model = MODEL[size]()
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, self.model.name, 'cp.ckpt')
        self.checkpoint_path = checkpoint_path
        
        if load:
            self.load_weights()

        self.print_template = 'Epoch {}/{}, ClsLoss: {:.4f}, TotalLoss: {:.4f}, Accuracy: {:.3f}'

        self._loss_ratio = {'cls': 1., 'box': 0.5, 'mark': 0.5}
        if size == 48:
            self._loss_ratio['mark'] = 1.

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def save_weights(self):
        self.model.save_weights(self.checkpoint_path)

    def load_weights(self):
        self.model.load_weights(self.checkpoint_path)

    @tf.function
    def train_step(self, dataset):
        imgs, clss, box, marks = next(dataset)
        with tf.GradientTape() as tape:
            p_clss, p_box, p_marks = self.model(imgs)
            cls_loss = self.model.classify_loss(clss, p_clss)
            box_loss = self.model.bbox_loss(clss, box, p_box)
            mark_loss = self.model.landmark_loss(clss, marks, p_marks)

            total_loss = self._loss_ratio['cls'] * cls_loss + self._loss_ratio['box'] * box_loss + self._loss_ratio['mark'] * mark_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return cls_loss, total_loss


    @tf.function
    def test_epoch(self, dataset):
        imgs, clss, *_ = next(dataset)

        p_clss = self.model.detect(imgs)
        acc = self.model.acc(clss, p_clss)
        
        return acc

    def train(self, dataset, steps=500, epochs=500, checkpoint_path=None, load=False):
        dataset = iter(dataset)

        bar = tqdm(range(epochs))
        for epoch in bar:
            for step in range(steps):
                cls_loss, total_loss = self.train_step(dataset)

            acc = self.test_epoch(dataset)

            self.save_weights()

            bar.set_description(self.print_template.format(epoch+1, epochs, cls_loss, total_loss, acc))



if __name__ == '__main__':

    # size = 12

    # dataset = DateSet(size)

    # train_ = Train(size, load=True)

    # train_.train(dataset.get_dataset())
    # del train_
    # del dataset

    # Train(24, load=True).train(DateSet(24).get_dataset())

    Train(48, load=True).train(DateSet(48).get_dataset())

    # train = Train('rnet', load=False)
    # train.train(dataset.dataset_for_other(24))



