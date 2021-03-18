import argparse
import sys
import tensorflow as tf

from logzero import setup_logger
import logging

loglvl = dict(info=logging.INFO, debug=logging.DEBUG, warning=logging.WARNING)
parser = argparse.ArgumentParser()
parser.add_argument("--loglvl", "-lvl", help="Set level of logging", default="info")


num_class = 10
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)


if __name__ == "__main__":

    args = parser.parse_args()
    logger = setup_logger(__file__, level=loglvl[args.loglvl])
    # Dowloading Mnist Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255, x_test / 255

    # Model Building
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(num_class, activation="softmax"))
    
    if args.loglvl == "debug":
        logger.debug(f"Model summary")
        model.build()
        model.summary()
        sys.exit(0)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test, y_test))
    logger.info("Done training!")