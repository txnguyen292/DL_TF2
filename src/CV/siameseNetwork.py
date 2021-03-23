import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random

# Prepare the dataset

def create_pair(x: np.ndarray, digit_indices: np.ndarray) -> np.ndarray, np.ndarray:
    """Positive and Negative pair creation.
    Alternates between positive and negative pairs

    Args:
        x (np.ndarray): input date
        digit_indices (np.ndarray): corresponding indices

    Returns:
        np.ndarray, np.ndarray: [description]
    """

    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices