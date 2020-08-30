'''
Author: your name
Date: 2020-08-11 16:17:08
LastEditTime: 2020-08-20 18:02:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \MTCNN-TensorFlow-2-master\mtcnn\model.py
'''
import tensorflow as tf


class NetWork(tf.keras.Model):
    def __init__(self, keep_ratio=0.7):
        super().__init__()
        assert 0 < keep_ratio <= 1, "什么玩意"
        self.keep_ratio = keep_ratio
       
    # @tf.function
    def classify_loss(self, y, y_):
        y_ = tf.squeeze(y_)
        y = tf.squeeze(y)

        keep = tf.where(tf.not_equal(y, -1))

        y_ = tf.gather(y_, keep[:, 0])
        y = tf.gather(y, keep[:, 0])
        y = tf.where(tf.equal(y, 0), y, tf.ones_like(y))


        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)

        top_k = tf.cast(tf.cast(tf.shape(y)[0], tf.float32)*self.keep_ratio, tf.int32)

        loss = tf.math.top_k(loss, top_k, False)[0]

        return tf.reduce_mean(loss)

    # @tf.function
    def bbox_loss(self, clss, y, y_):
        y_ = tf.squeeze(y_)
        keep = tf.where(tf.not_equal(clss, 0))
        y = tf.squeeze(y)

        y_ = tf.gather(y_, keep[:, 0])
        y = tf.gather(y, keep[:, 0])

        loss = tf.math.square(y-y_)
        loss = tf.reduce_mean(loss, axis=1)
        
        top_k = tf.cast(tf.cast(tf.shape(y)[0], tf.float32)*self.keep_ratio, tf.int32)

        loss = tf.math.top_k(loss, top_k, False)[0]

        return tf.reduce_mean(loss)

    # @tf.function
    def landmark_loss(self, clss, y, y_):
        y_ = tf.squeeze(y_)
        keep = tf.where(tf.equal(clss, -2))
        y = tf.squeeze(y)

        y_ = tf.gather(y_, keep[:, 0])
        y = tf.gather(y, keep[:, 0])

        loss = tf.square(y-y_)
        loss = tf.reduce_mean(loss, axis=1)
        
        top_k = tf.cast(tf.cast(tf.shape(y)[0], tf.float32)*self.keep_ratio, tf.int32)

        loss = tf.math.top_k(loss, top_k, False)[0]

        return tf.reduce_mean(loss)   

    def acc(self, y, y_):
        y_ = tf.squeeze(y_)
        y = tf.squeeze(y)

        keep = tf.where(tf.not_equal(y, -1))
        y_ = tf.gather(y_, keep[:, 0])
        y = tf.gather(y, keep[:, 0])

        y_ = tf.argmax(tf.squeeze(y_), axis=1)

        y = tf.where(tf.equal(y, 0), y, tf.ones_like(y, dtype=tf.int64))

        res = tf.cast(tf.equal(y_, y), dtype=tf.float32)
        return tf.reduce_mean(res)


    def detect(self, inputs):
        x = self.layers_(inputs)
        return self.face_class(x)


class PNet(NetWork):
    def __init__(self):
        super().__init__()
        self.layers_ = tf.keras.Sequential([
            tf.keras.layers.Conv2D(10, (3, 3), input_shape=(None, None, 3), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),

            tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
        ])

        self.face_class = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax')
        self.box_reg = tf.keras.layers.Conv2D(4, (1, 1))
        self.landmark_reg = tf.keras.layers.Conv2D(10, (1, 1))

        # self.compile(optimizer=self.optimizer, loss=[self.classify_loss, self.bbox_loss, self.landmark_loss],
        #              metrics=[[self.acc], [], []], loss_weights=[1, 0.5, 0.5])

    def call(self, inputs):
        x = self.layers_(inputs)
        cls = self.face_class(x)
        bbox = self.box_reg(x)
        landmark = self.landmark_reg(x)

        return cls, bbox, landmark

    def predict(self, inputs):
        x = self.layers_(inputs)
        cls = self.face_class(x)
        bbox = self.box_reg(x)

        return cls.numpy(), bbox.numpy()


class RNet(NetWork):
    def __init__(self):
        super().__init__()
        self.layers_ = tf.keras.Sequential([
            tf.keras.layers.Conv2D(28, (3, 3), input_shape=(24, 24, 3), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2)),

            tf.keras.layers.Conv2D(48, (3, 3), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2)),

            tf.keras.layers.Conv2D(64, (2, 2), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),

        ])
        self.face_class = tf.keras.layers.Dense(2, activation='softmax')
        self.box_reg = tf.keras.layers.Dense(4)
        self.landmark_reg = tf.keras.layers.Dense(10)

        # self.compile(optimizer=self.optimizer, loss=[self.classify_loss, self.bbox_loss, self.landmark_loss],
        #              metrics=[[self.acc], [], []], loss_weights=[1, 0.5, 0.5])

    def call(self, inputs):
        x = self.layers_(inputs)
        cls = self.face_class(x)
        bbox = self.box_reg(x)
        landmark = self.landmark_reg(x)

        return cls, bbox, landmark

    def predict(self, inputs):
        x = self.layers_(inputs)
        cls = self.face_class(x)
        bbox = self.box_reg(x)

        return cls.numpy(), bbox.numpy()


class ONet(NetWork):
    def __init__(self):
        super().__init__()
        self.layers_ = self.layers_ = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=(48, 48, 3), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(128, (2, 2), kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),

        ])
        self.face_class = tf.keras.layers.Dense(2, activation='softmax')
        self.box_reg = tf.keras.layers.Dense(4)
        self.landmark_reg = tf.keras.layers.Dense(10)

        # self.compile(optimizer=self.optimizer, loss=[self.classify_loss, self.bbox_loss, self.landmark_loss],
        #              metrics=[[self.acc], [], []], loss_weights=[1, 0.5, 1])

    def call(self, inputs):
        x = self.layers_(inputs)
        cls = self.face_class(x)
        bbox = self.box_reg(x)
        landmark = self.landmark_reg(x)

        return cls, bbox, landmark


if __name__ == '__main__':
    model = PNet()
    print(model.layers_)
