
from PIL import Image
import numpy as np
import tensorflow as tf
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

def process_observation(observation):
    assert observation.ndim == 3  # (height, width, channel)
    img = Image.fromarray(observation)
    img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
    processed_observation = np.array(img)
    assert processed_observation.shape == INPUT_SHAPE
    return processed_observation.astype('uint8') 

def process_state_batch(batch):
    # We could perform this processing step in `process_observation`. In this case, however,
    # we would need to store a `float32` array instead, which is 4x more memory intensive than
    # an `uint8` array. This matters if we store 1M observations.
    processed_batch = batch.astype('float32') / 255.
    return processed_batch


