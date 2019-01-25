'''
# Variational AutoEncoder Trainer
# Author: Zhihui Lu
# Date: 2018/10/17
'''
import tensorflow as tf
import numpy as np
import os, random
from tqdm import tqdm
import dataIO as io
from network import *
from model import Variational_Autoencoder
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          #for windows
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("train_data_txt", 'F:/data_info/VAE_liver/set_5/TFrecord/fold_1/train.txt', "train data txt")
    flags.DEFINE_string("outdir", 'G:/experiment_result/liver/VAE/set_5/down/64/alpha_0.1/fold_1/beta_1', "outdir")
    flags.DEFINE_string("gpu_index", "0", "GPU-index")
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("batch_size", 12, "batch size")
    flags.DEFINE_integer("num_iteration", 12001, "number of iteration")
    flags.DEFINE_integer("save_loss_step", 150, "step of save loss")
    flags.DEFINE_integer("save_model_step", 150, "step of save model and validation")
    flags.DEFINE_integer("shuffle_buffer_size", 200, "buffer size of shuffle")
    flags.DEFINE_integer("latent_dim", 4, "latent dim")
    flags.DEFINE_list("image_size", [56, 72, 88, 1], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard'))
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'model'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'model'))

    # read list
    train_data_list = io.load_list(FLAGS.train_data_txt)
    # shuffle list
    random.shuffle(train_data_list)

    # load train data
    train_set = tf.data.Dataset.list_files(train_data_list)
    train_set = train_set.apply(
        tf.contrib.data.parallel_interleave(lambda x: tf.data.TFRecordDataset(x, compression_type = 'GZIP'),
                                            cycle_length = os.cpu_count()))
    train_set = train_set.map(lambda x: utils._parse_function(x, image_size=FLAGS.image_size),
                              num_parallel_calls=os.cpu_count())
    # train_set = train_set.cache()
    train_set = train_set.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    train_set = train_set.repeat()
    train_set = train_set.batch(FLAGS.batch_size)
    train_iter = train_set.make_one_shot_iterator()
    train_data = train_iter.get_next()

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config(index=FLAGS.gpu_index)) as sess:
        # # set network

        kwargs = {
            'sess': sess,
            'outdir': FLAGS.outdir,
            'beta': FLAGS.beta,
            'latent_dim': FLAGS.latent_dim,
            'batch_size': FLAGS.batch_size,
            'image_size': FLAGS.image_size,
            'encoder': encoder_resblock_bn,
            'decoder': decoder_resblock_bn,
            'downsampling': down_sampling,
            'upsampling': up_sampling,
            'learning_rate': 1e-4,
            'is_training': True,
            'is_down': False
        }
        VAE = Variational_Autoencoder(**kwargs)

        # print parmeters
        utils.cal_parameter()

        # prepare tensorboard
        writer_train = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'train'), sess.graph)
        writer_rec = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'train_rec'))
        writer_kl = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'train_kl'))

        value_loss = tf.Variable(0.0)
        tf.summary.scalar("loss", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        # # training
        tbar = tqdm(range(FLAGS.num_iteration), ascii=True)
        epoch_train_loss = []
        epoch_kl_loss = []
        epoch_rec_loss = []
        for i in tbar:
            train_data_batch = sess.run(train_data)

            train_loss, rec_loss, kl_loss = VAE.update(train_data_batch)
            epoch_train_loss.append(train_loss)
            epoch_kl_loss.append(kl_loss)
            epoch_rec_loss.append(rec_loss)

            if i % FLAGS.save_loss_step is 0:
                s = "Loss: {:.4f}, kl_loss: {:.4f}, rec_loss: {:.4f}"\
                    .format(np.mean(epoch_train_loss), np.mean(epoch_kl_loss), np.mean(epoch_rec_loss))
                tbar.set_description(s)

                summary_train_loss = sess.run(merge_op, {value_loss: train_loss})
                summary_rec_loss = sess.run(merge_op, {value_loss: rec_loss})
                summary_kl_loss = sess.run(merge_op, {value_loss: kl_loss})
                writer_train.add_summary(summary_train_loss, i)
                writer_rec.add_summary(summary_rec_loss, i)
                writer_kl.add_summary(summary_kl_loss, i)

                epoch_train_loss.clear()
                epoch_kl_loss.clear()
                epoch_rec_loss.clear()

            if i % FLAGS.save_model_step is 0:
                # save model
                VAE.save_model(i)

if __name__ == '__main__':
    main()