'''
# Variational AutoEncoder Model
# Author: Zhihui Lu
# Date: 2018/10/17
# Reference: "https://github.com/wuga214/IMPLEMENTATION_Variational-Auto-Encoder"
'''

import tensorflow as tf
import os
import numpy as np

class Variational_Autoencoder(object):

    def __init__(self, sess, outdir, beta, latent_dim, batch_size, image_size, encoder, decoder,
                 downsampling, upsampling, is_training=True, is_down=False, learning_rate=1e-4):

        self._sess = sess
        self._outdir = outdir
        self._beta = beta
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._image_size = image_size
        self._learning_rate = learning_rate
        self._encoder = encoder
        self._decoder = decoder
        self._down = downsampling
        self._up = upsampling
        self._is_training = is_training
        self._is_down = is_down

        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope('input'):
            if len(self._image_size) ==1:
                self.input = tf.placeholder(tf.float32, shape=[None, int(np.prod(self._image_size))])
            if len(self._image_size) ==4:
                self.input = tf.placeholder(tf.float32, shape = [None, self._image_size[0], self._image_size[1],
                                                                 self._image_size[2], self._image_size[3]])
            if self._is_down==True:
                self.input_down = self._down(self.input)
            else:
                self.input_down = self.input

        with tf.variable_scope('encoder'):
            encoded, self._shape_size = self._encoder(self.input_down, self._latent_dim, self._is_training)
            # tf.summary.histogram("encoder", encoded)      #histogram

        with tf.variable_scope('latent_space'):
            with tf.variable_scope('latent_mean'):
                self.mean = encoded[:, :self._latent_dim]

            with tf.variable_scope('latent_sigma'):
                stddev = encoded[:, self._latent_dim:]
                # stddev = tf.sqrt(tf.exp(logvar))

            epsilon = tf.random_normal([self._batch_size, self._latent_dim], mean=0., stddev=1.0)
            with tf.variable_scope('latent_sample'):
                self.z = self.mean + stddev * epsilon

        with tf.variable_scope('decoder'):
            self.decode_result = self._decoder(self.z, self._batch_size, self._shape_size, self._image_size, self._is_training)
            # tf.summary.histogram("decoder", decoded)      #histogram

        with tf.variable_scope('loss'):
            with tf.variable_scope('reconstruction-loss'):
                # rec_loss = self._rec_loss(self.decode_result, self.input_down, feature_size=int(np.prod(self._image_size) / 8))
                rec_loss = self._rec_loss(self.decode_result, self.input_down, feature_size=int(np.prod(self._image_size)))

                self._rec_loss = tf.reduce_mean(rec_loss)
                # self._rec_loss = rec_loss

            with tf.variable_scope('kl-divergence'):
                kl = self._kl_diagnormal_stdnormal(self.mean, stddev)
                self._kl_loss = tf.reduce_mean(kl)

            self._loss = self._rec_loss + self._kl_loss * self._beta

        self._val_loss = self._loss
        self._val_kl = self._kl_loss
        self._val_rec = self._rec_loss

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate, beta1=0.5, beta2=0.9)

        with tf.variable_scope('training-step'):
            self._train = optimizer.minimize(self._loss)

        self.saver = tf.train.Saver(max_to_keep=None)
        init = tf.initializers.global_variables()
        self._sess.run(init)

    @staticmethod
    def _kl_diagnormal_stdnormal(mu, z_sigma, bias=1e-5):
        kl = - 0.5 * tf.reduce_sum(1 + tf.log(tf.square(z_sigma) + bias) - tf.square(mu) - tf.square(z_sigma), axis=-1)
        return kl

    @staticmethod
    def _rec_loss(pred, ground_truth, feature_size):
        pred_reshaped = tf.reshape(pred, [-1, feature_size])
        ground_truth_reshaped = tf.reshape(ground_truth, [-1, feature_size])
        se = tf.losses.mean_squared_error(pred_reshaped, ground_truth_reshaped) * feature_size
        # se = tf.losses.mean_squared_error(pred_reshaped, ground_truth_reshaped)
        # se = tf.losses.absolute_difference(pred_reshaped, ground_truth_reshaped) * 500
        return se

    def update(self, X):
        _, loss, loss_rec, loss_kl = self._sess.run([self._train, self._loss, self._rec_loss, self._kl_loss], feed_dict = {self.input : X})
        return loss, loss_rec, loss_kl

    def save_model(self, index):
        save = self.saver.save(self._sess, os.path.join(self._outdir, 'model' , 'model_{}'.format(index)))
        return save

    def restore_model(self, path):
        self.saver.restore(self._sess, path)

    def validation(self, X):
        val_loss, val_rec, val_kl = self._sess.run([self._val_loss, self._val_rec, self._val_kl], feed_dict = {self.input : X})
        return val_loss, val_rec, val_kl

    def reconstruction_image(self, X):
        hidden = self._sess.run(self.mean, feed_dict = {self.input: X})
        predict = self._sess.run(self.decode_result, feed_dict = {self.z: hidden})
        # predict = self._up(tf.convert_to_tensor(predict)).eval()
        return predict

    def plot_latent(self, X):
        hidden = self._sess.run(self.mean, feed_dict = {self.input: X})
        return hidden

    def generate_sample(self,z):
        x = self._sess.run(self.decode_result, feed_dict = {self.z: z})
        # x = self._up(tf.convert_to_tensor(x)).eval()
        return  x