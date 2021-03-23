from config import CONFIG
import logging
from logzero import setup_logger
from typing import List, Tuple, Union, Iterable

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Layer

# Implement Model subclasses

class IdentityBlock(tf.keras.Model):
    """Resnet Identity block
    """

    def __init__(self, filters: int, kernel_size:Union[int, Iterable[int]]) -> None:
        super(IdentityBlock, self).__init__(name="")
        
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation("relu")
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.add([x, input_tensor])
        x = self.act(x)
        return x
    
class ResNet(tf.keras.Model):
    """Mini Resnet"""
    def __init__(self, num_classes):

        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("relu")
        self.max_pool = tf.keras.layers.MaxPool2D((3, 3))

        # Use the identity blocks
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)
        x = self.global_pool(x)

        return self.classifier(x)

def preprocess(features):
    return tf.cast(features["image"], tf.float32) / 255., features["label"]

if __name__ == "__main__":

    logger = setup_logger(__file__, level=logging.INFO)
    # id = IdentityBlock(64, 3)
    # id.build(input_shape=(32, 256, 256, 1))
    # logger.debug(f"{id.summary()}")
    data_dir = CONFIG.data / "external"
    # print(list(data_dir.glob("*")))
    # logger.info()
    # resnet = ResNet(10)
    # resnet.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # logger.info("Download mnist dataset from tfds")
    dataset = tfds.load("mnist", split=tfds.Split.TRAIN, data_dir=str(data_dir))
    # dataset = dataset.map(preprocess).batch(32)
    # logger.info("Start training...")
    # history = resnet.fit(dataset, epochs=1)
    # logger.info("Done!")