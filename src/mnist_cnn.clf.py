import argparse
import sys
from typing import Tuple, Any, Dict, Union
from numpy.lib.histograms import histogram
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import logging
from logzero import setup_logger

Tensor = Any
loglvl = dict(info=logging.INFO, debug=logging.DEBUG, warning=logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--loglvl", "-lvl", help="Set logging level", default="info")
num_classes = 10
img_rows, img_cols, img_ch = 28, 28, 1
input_shape = (img_rows, img_cols, img_ch)

class SimpleConvolutionalLayer(tf.keras.layers.Layer):
    """CNN Layer"""

    def __init__(self, num_kernels: int=32, kernel_size: Tuple[int]=(3, 3), 
                strides: Tuple[int]=(1, 1), use_bias: bool=True) -> None:
        """

        Args:
            num_kernels (int, optional): Number of kernels for the convolution. Defaults to 32.
            kernel_size (Tuple[int], optional): Kernel size (H x W). Defaults to (3, 3).
            strides (Tuple[int], optional): Vertical and horizontal stride as list. Defaults to (1, 1).
            use_bias (bool, optional): Flag to add a bias after convolution before activation. Defaults to True.
        """
        # Initializing subclass
        super().__init__()
        
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias

    def build(self, input_shape: Tuple[int]) -> None:
        """Build the layer, initializing its parameters according to
        the input shape.
        This function will be internally called the first time the layer is used,
        though it can also be manually called

        Args:
            input_shape (Tuple[int]): Input shape the layer will receive (B x H x W x C)
        """

        num_input_channels = input_shape[-1]

        kernels_shape = (*self.kernel_size, num_input_channels, self.num_kernels)
        # Initialize the filters with Glorot distribution
        glorot_uni_initializer = tf.initializers.GlorotUniform()
        self.kernels = self.add_weight(name="kernels",
                                       shape=kernels_shape,
                                       initializer=glorot_uni_initializer,
                                       trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name="bias",
                                        shape=(self.num_kernels,),
                                        initializer="random_normal",
                                        trainable=True)
        
        def call(self, inputs: Tensor):
            """Call the layer and perform its operations on the input tensors

            Args:
                inputs ([type]): [description]
            """
            z = tf.nn.conv2d(inputs, self.kernels, strides=[1, *self.strides, 1], padding="VALID")
            if self.use_bias:
                z = z + self.bias
            return tf.nn.relu(z)

        def get_config(self) -> Dict[int, Union[int, str]]:
            """
            Helper function to define the layer and its parameters.

            Returns:
                Dict[int, Union[int, str]]: model config
            """
            return {"num_kernels": self.num_kernels,
                    "kernel_size": self.kernel_size,
                    "strides": self.strides,
                    "use_bias": self.use_bias}
class Lenet5(Model):
    """Lenet5

    Args:
        Model ([type]): tf.keras.Model
    """

    def __init__(self, num_classes: int) -> None:
        super(Lenet5, self).__init__()
        self.conv1 = Conv2D(6, kernel_size=(5, 5), padding="same", activation="relu")
        self.conv2 = Conv2D(16, kernel_size=(5, 5), activation="relu")
        self.max_pool = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation="relu")
        self.dense2 = Dense(84, activation="relu")
        self.dense3 = Dense(num_classes, activation="softmax")

    def call(self, inputs) -> Tensor:
        """Call the layers and perform their operations on the input tensors

        Args:
            inputs ([type]): [description]
        """
        x = self.max_pool(self.conv1(inputs))
        x = self.max_pool(self.conv2(x))
        x = self.flatten(x)
        x = self.dense3(self.dense2(self.dense1(x)))
        return x
            
if __name__ == "__main__":

    args = parser.parse_args()
    logger = setup_logger(__file__, level=loglvl[args.loglvl])
    logger.info("Download mnist dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255, x_test / 255
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)
    model = Lenet5(num_classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    if args.loglvl == "debug":
        logger.debug(f"X_train, X_test shape: {x_train.shape, x_test.shape}")
        batched_input_shape = tf.TensorShape((None, *input_shape))
        model.build(input_shape=batched_input_shape)
        model.summary()
        sys.exit(0)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss"),
        tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1, write_graph=True)
    ]

    history = model.fit(x_train, y_train, batch_size=32, epochs=80, 
                        validation_data=(x_test, y_test), verbose=2, callbacks=callbacks)