import tensorflow as tf
import numpy as np

# # load tfrecord function
def _parse_function(record, image_size=[512, 512, 1]):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    image = parsed_features['img_raw']
    image = tf.reshape(image, image_size)
    return image

# session config
def config(index = "0"):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=index , # specify GPU number
            allow_growth=True
        )
    )
    return config

# calculate total parameters
def cal_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return print('Total params: %d ' % total_parameters)

# calculate jaccard
def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.bitwise_and(im1, im2).sum()) / np.double(np.bitwise_or(im1, im2).sum())
