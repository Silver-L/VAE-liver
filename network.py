'''
# Network Architecture 3 dim
# Author: Zhihui Lu
# Date: 2018/12/18
'''

import tensorflow as tf
import numpy as np


# # Dense with batch normalization
def encoder_dense(input, latent_dim, is_training=True):
    encoder = dense_layer(input, unit=200, activation=True, is_training=is_training)
    encoder = tf.layers.dropout(encoder, rate=0.5)
    encoder = dense_layer(encoder, unit=150, activation=True, is_training=is_training)
    encoder = tf.layers.dropout(encoder, rate=0.5)
    encoder = dense_layer(encoder, unit=100, activation=True, is_training=is_training)
    encoder = tf.layers.dropout(encoder, rate=0.5)
    encoder = dense_layer(encoder, unit=50, activation=True, is_training=is_training)
    encoder = tf.layers.dropout(encoder, rate=0.5)
    # encoder = dense_layer(encoder, unit=70, activation=True, is_training=is_training)
    # encoder = tf.layers.dropout(encoder, rate=0.5)
    # encoder = dense_layer(encoder, unit=40, activation=True, is_training=is_training)
    # encoder = tf.layers.dropout(encoder, rate=0.5)
    encoder = dense_layer(encoder, unit=20, activation=True, is_training=is_training)

    shape_size = tuple(encoder.get_shape().as_list())

    encoder = tf.layers.dense(encoder, 2 * latent_dim)
    return encoder, shape_size

def decoder_dense(input, batch_size, shape_before_flatten, image_size, is_training=True):
    decoder = dense_layer(input, unit=20, activation=True, is_training=is_training)
    decoder = tf.layers.dropout(decoder, rate=0.5)
    decoder = dense_layer(decoder, unit=50, activation=True, is_training=is_training)
    decoder = tf.layers.dropout(decoder, rate=0.5)
    decoder = dense_layer(decoder, unit=100, activation=True, is_training=is_training)
    decoder = tf.layers.dropout(decoder, rate=0.5)
    decoder = dense_layer(decoder, unit=150, activation=True, is_training=is_training)
    # decoder = tf.layers.dropout(decoder, rate=0.5)
    # decoder = dense_layer(decoder, unit=140, activation=True, is_training=is_training)
    # decoder = tf.layers.dropout(decoder, rate=0.5)
    # decoder = dense_layer(decoder, unit=170, activation=True, is_training=is_training)
    # decoder = tf.layers.dropout(decoder, rate=0.5)
    decoder = dense_layer(decoder, unit=200, activation=True, is_training=is_training)
    decoder = tf.layers.dense(decoder, units=np.prod(image_size))

    # decoder = tf.layers.dense(decoder, units=np.prod(image_size), activation=tf.nn.sigmoid)
    return decoder

def dense_layer(input, unit, activation=True, is_training=True):
    output = tf.layers.dense(input, units=unit)
    # output = tf.layers.batch_normalization(output, training=is_training)
    if activation==True:
        output = tf.nn.relu(output)
    else:
        output = output
    return output


# # Convolution with batch normalization
def encoder_conv_with_bn(input, latent_dim, is_training=True):
    encoder = conv_with_bn(input, filters=32, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=is_training)
    encoder = conv_with_bn(encoder, filters=64, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=is_training)
    encoder = conv_with_bn(encoder, filters=128, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=is_training)
    encoder = conv_with_bn(encoder, filters=256, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=is_training)

    shape_before_flatten = tuple(encoder.get_shape().as_list())

    # encoder to latent space
    encoder = tf.layers.flatten(encoder)
    encoder = tf.layers.dense(encoder, 2 * latent_dim)
    return encoder, shape_before_flatten

def decoder_conv_with_bn(input, batch_size, shape_before_flatten, image_size, is_training=True):
    # latent space to decoder
    decoder = tf.layers.dense(input, units=np.prod(shape_before_flatten[1:]), activation=tf.nn.relu)
    decoder = tf.reshape(decoder, [batch_size, shape_before_flatten[1],
                                   shape_before_flatten[2], shape_before_flatten[3], shape_before_flatten[4]])

    decoder = deconv_with_bn(decoder, filters=128, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=is_training)
    decoder = deconv_with_bn(decoder, filters=64, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=is_training)
    decoder = deconv_with_bn(decoder, filters=32, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=is_training)
    decoder = tf.layers.conv3d_transpose(decoder, filters=1, kernel_size=(5,5,5), strides=(2,2,2), padding='same', activation=tf.nn.sigmoid)
    return decoder

def conv_with_bn(input, filters, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=True):
    output = tf.layers.conv3d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    output = batch_norm(output, is_training=is_training)
    output = tf.nn.relu(output)
    return output

def deconv_with_bn(input, filters, kernel_size=(5,5,5), strides=(2,2,2), padding='same', is_training=True):
    output = tf.layers.conv3d_transpose(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    output = batch_norm(output, is_training=is_training)
    output = tf.nn.relu(output)
    return output


# # Convolution with group normalization
def encoder_conv_with_gn(input, latent_dim, is_training=True):
    encoder = conv_with_gn(input, filters=32, kernel_size=(5,5,5), strides=(2,2,2), padding='same')
    encoder = conv_with_gn(encoder, filters=64, kernel_size=(5,5,5), strides=(2,2,2), padding='same')
    encoder = conv_with_gn(encoder, filters=128, kernel_size=(5,5,5), strides=(2,2,2), padding='same')
    encoder = conv_with_gn(encoder, filters=256, kernel_size=(5,5,5), strides=(2,2,2), padding='same')

    shape_before_flatten = tuple(encoder.get_shape().as_list())

    # encoder to latent space
    encoder = tf.layers.flatten(encoder)
    encoder = tf.layers.dense(encoder, 2 * latent_dim)
    return encoder, shape_before_flatten

def decoder_conv_with_gn(input, batch_size, shape_before_flatten, image_size, is_training=True):
    # latent space to decoder
    decoder = tf.layers.dense(input, units=np.prod(shape_before_flatten[1:]), activation=tf.nn.relu)
    decoder = tf.reshape(decoder, [batch_size, shape_before_flatten[1],
                                   shape_before_flatten[2], shape_before_flatten[3], shape_before_flatten[4]])

    decoder = deconv_with_gn(decoder, filters=128, kernel_size=(5,5,5), strides=(2,2,2), padding='same')
    decoder = deconv_with_gn(decoder, filters=64, kernel_size=(5,5,5), strides=(2,2,2), padding='same')
    decoder = deconv_with_gn(decoder, filters=32, kernel_size=(5,5,5), strides=(2,2,2), padding='same')
    decoder = tf.layers.conv3d_transpose(decoder, filters=1, kernel_size=(5,5,5), strides=(2,2,2), padding='same', activation=tf.nn.sigmoid)
    return decoder

def conv_with_gn(input, filters, kernel_size=(5,5,5), strides=(2,2,2), padding='same'):
    with tf.variable_scope('conv_gama_beta{}'.format(filters)) :
        gamma = tf.get_variable('gamma', [1, 1, 1, 1, filters], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, 1, filters], initializer=tf.constant_initializer(0.0))

    output = tf.layers.conv3d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    output = GroupNorm(output, gama=gamma, beta=beta, G=32)
    output = tf.nn.relu(output)
    return output

def deconv_with_gn(input, filters, kernel_size=(5,5,5), strides=(2,2,2), padding='same'):
    with tf.variable_scope('decov_gama_beta{}'.format(filters)) :
        gamma = tf.get_variable('gamma', [1, 1, 1, 1, filters], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, 1, filters], initializer=tf.constant_initializer(0.0))

    output = tf.layers.conv3d_transpose(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    output = GroupNorm(output, gama=gamma, beta=beta, G=32)
    output = tf.nn.relu(output)
    return output

def GroupNorm(x, gama, beta, G, eps=1e-5):
    N, D, H, W, C = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [N, D, H, W, G, C // G])
    mean, var = tf.nn.moments(x, [1, 2, 3, 5], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, D, H, W, C])
    return x * gama + beta


# resblock with batch normalization
def encoder_resblock_bn(input, latent_dim, is_training=True):
    encoder = tf.layers.conv3d(input, kernel_size=7, filters=64, strides=2, padding='same')
    encoder = tf.keras.layers.MaxPool3D(pool_size=3, strides=2, padding='same')(encoder)
    encoder = resblock_with_bn(encoder, 64, is_training=is_training)
    encoder = resblock_with_bn(encoder, 64, is_training=is_training)
    encoder = resblock_with_bn(encoder, 128, downsample=True, is_training=is_training)
    encoder = resblock_with_bn(encoder, 128, is_training=is_training)
    # encoder = resblock_with_bn(encoder, 256, downsample=True, is_training=is_training)
    # encoder = resblock_with_bn(encoder, 256, is_training=is_training)

    shape_before_flatten = tuple(encoder.get_shape().as_list())
    encoder = tf.layers.dense(encoder, units=1000)

    # encoder to latent space
    output = tf.layers.flatten(encoder)
    output = tf.layers.dense(output, 2 * latent_dim)
    return output, shape_before_flatten

def decoder_resblock_bn(input, batch_size, shape_before_flatten, image_size, is_training=True):
    decoder = tf.layers.dense(input, units=np.prod(shape_before_flatten[1:]), activation=tf.nn.relu)
    decoder = tf.reshape(decoder, [batch_size, shape_before_flatten[1],
                                   shape_before_flatten[2], shape_before_flatten[3], shape_before_flatten[4]])

    # decoder = deconv_with_bn(decoder, filters=256, kernel_size=5, strides=2, padding='same', is_training=is_training)
    decoder = deconv_with_bn(decoder, filters=128, kernel_size=5, strides=2, padding='same', is_training=is_training)
    decoder = deconv_with_bn(decoder, filters=64, kernel_size=5, strides=2, padding='same', is_training=is_training)
    decoder = deconv_with_bn(decoder, filters=32, kernel_size=5, strides=2, padding='same', is_training=is_training)
    output = tf.layers.conv3d(decoder, filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)
    return output

def resblock_with_bn(x_init, channels, use_bias=True, downsample=False, is_training=True):
    x = batch_norm(x_init, is_training=is_training)
    x = tf.nn.relu(x)
    if downsample:
        x = tf.layers.conv3d(x, filters=channels, kernel_size=3, strides=2, use_bias=use_bias, padding='same')
        x_init = tf.layers.conv3d(x_init, filters=channels, kernel_size=1, strides=2, use_bias=use_bias, padding='same')

    else:
        x = tf.layers.conv3d(x, filters=channels, kernel_size=3, strides=1, use_bias=use_bias, padding='same')
    x = batch_norm(x, is_training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv3d(x, filters=channels, kernel_size=3, strides=1, use_bias=use_bias, padding='same')
    return x + x_init


# resblock with group normalization
def encoder_resblock_gn(input, latent_dim, is_training=True):
    encoder = tf.layers.conv3d(input, kernel_size=7, filters=64, strides=2, padding='same')
    encoder = tf.keras.layers.MaxPool3D(pool_size=3, strides=2, padding='same')(encoder)
    encoder = resblock_with_gn(encoder, 64)
    encoder = resblock_with_gn(encoder, 64)
    encoder = resblock_with_gn(encoder, 128, downsample=True)
    encoder = resblock_with_gn(encoder, 128)
    encoder = resblock_with_gn(encoder, 256, downsample=True)
    encoder = resblock_with_gn(encoder, 256)

    shape_before_flatten = tuple(encoder.get_shape().as_list())
    encoder = tf.layers.dense(encoder, units=1000)
    encoder = tf.layers.dropout(encoder, rate=0.5)

    # encoder to latent space
    output = tf.layers.flatten(encoder)
    output = tf.layers.dense(output, 2 * latent_dim)
    return output, shape_before_flatten

def decoder_resblock_gn(input, batch_size, shape_before_flatten, image_size, is_training=True):
    decoder = tf.layers.dense(input, units=np.prod(shape_before_flatten[1:]), activation=tf.nn.relu)
    decoder = tf.reshape(decoder, [batch_size, shape_before_flatten[1],
                                   shape_before_flatten[2], shape_before_flatten[3], shape_before_flatten[4]])

    decoder = deconv_with_gn(decoder, filters=256, kernel_size=5, strides=2, padding='same')
    decoder = deconv_with_gn(decoder, filters=128, kernel_size=5, strides=2, padding='same')
    decoder = deconv_with_gn(decoder, filters=64, kernel_size=5, strides=2, padding='same')
    decoder = deconv_with_gn(decoder, filters=32, kernel_size=5, strides=2, padding='same')
    output = tf.layers.conv3d(decoder, filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)
    return output

def resblock_with_gn(x_init, channels, use_bias=True, downsample=False):
    with tf.variable_scope('res_gama_beta{}'.format(channels)) :
        gamma = tf.get_variable('gamma', [1, 1, 1, 1, channels], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, 1, channels], initializer=tf.constant_initializer(0.0))

    if downsample:
        x = tf.layers.conv3d(x_init, filters=channels, kernel_size=3, strides=2, use_bias=use_bias, padding='same')
        x_init = tf.layers.conv3d(x_init, filters=channels, kernel_size=1, strides=2, use_bias=use_bias, padding='same')
    else:
        x = tf.layers.conv3d(x_init, filters=channels, kernel_size=3, strides=1, use_bias=use_bias, padding='same')
    x = GroupNorm(x,gamma, beta, G=32)
    x = tf.nn.relu(x)
    x = tf.layers.conv3d(x, filters=channels, kernel_size=3, strides=1, use_bias=use_bias, padding='same')
    x = GroupNorm(x,gamma, beta, G=32)
    x = x + x_init
    x = tf.nn.relu(x)
    return x


# # up/down sampling
def down_sampling(input, pool_size=(2,2,2)):
    output = tf.keras.layers.AveragePooling3D(pool_size=pool_size)(input)
    return output

def up_sampling(input, size=(2,2,2)):
    output = tf.keras.layers.UpSampling3D(size=size)(input)
    return output

def batch_norm(x, is_training=True):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training)
