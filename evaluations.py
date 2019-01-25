'''
# Variational AutoEncoder Evaluation
# Author: Zhihui Lu
# Date: 2018/12/18
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
    flags.DEFINE_string("val_data_txt", 'F:/data_info/VAE_liver/set_5/TFrecord/fold_1/val.txt', "validation data txt")
    flags.DEFINE_string("model_dir", 'G:/experiment_result/liver/VAE/set_5/down/64/alpha_0.1/fold_1/beta_10/model', "dir of model")
    flags.DEFINE_string("outdir", 'G:/experiment_result/liver/VAE/set_5/down/64/alpha_0.1/fold_1/beta_10', "outdir")
    flags.DEFINE_string("gpu_index", "0", "GPU-index")
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_val", 76, "number of validation data")
    flags.DEFINE_integer("train_iteration", 12001, "number of training iteration")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("num_per_val", 150, "number per each validation(equal step of saving model)")
    flags.DEFINE_integer("latent_dim", 4, "latent dim")
    flags.DEFINE_list("image_size", [56, 72, 88, 1], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard'))

    # read list
    val_data_list = io.load_list(FLAGS.val_data_txt)

    # number of model
    num_of_model = FLAGS.train_iteration // FLAGS.num_per_val
    if FLAGS.train_iteration % FLAGS.num_per_val !=0:
        num_of_model += 1
    if FLAGS.train_iteration % FLAGS.num_per_val ==0:
        num_of_model -= 1

    # val_iter
    num_val_iter = FLAGS.num_of_val // FLAGS.batch_size
    if FLAGS.num_of_val % FLAGS.batch_size != 0:
        num_val_iter += 1

    # load validation data
    val_set = tf.data.TFRecordDataset(val_data_list, compression_type = 'GZIP')
    val_set = val_set.map(lambda x: utils._parse_function(x, image_size=FLAGS.image_size),
                          num_parallel_calls=os.cpu_count())
    val_set = val_set.repeat()
    val_set = val_set.batch(FLAGS.batch_size)
    val_iter = val_set.make_one_shot_iterator()
    val_data = val_iter.get_next()

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
            'is_training': False,
            'is_down': False
        }
        VAE = Variational_Autoencoder(**kwargs)

        # print parmeters
        utils.cal_parameter()

        # prepare tensorboard
        writer_val = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'val'))
        writer_val_rec = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'val_rec'))
        writer_val_kl = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'val_kl'))

        value_loss = tf.Variable(0.0)
        tf.summary.scalar("loss", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        # # validation
        tbar = tqdm(range(num_of_model), ascii=True)
        for i in tbar:
            VAE.restore_model(FLAGS.model_dir + '/model_{}'.format(i*FLAGS.num_per_val))

            val_loss_all = []
            val_rec_all = []
            val_kl_all = []
            for j in range(num_val_iter):
                val_data_batch = sess.run(val_data)
                val_loss, val_rec, val_kl = VAE.validation(val_data_batch)
                val_loss_all.append(val_loss)
                val_rec_all.append(val_rec)
                val_kl_all.append(val_kl)
            val_loss, val_rec, val_kl = np.mean(val_loss_all), np.mean(val_rec_all), np.mean(val_kl_all)
            s = "val: {:.4f}, val_rec: {:.4f}, val_kl: {:.4f} ".format(val_loss, val_rec, val_kl)
            tbar.set_description(s)

            summary_val = sess.run(merge_op, {value_loss: val_loss})
            summary_val_rec = sess.run(merge_op, {value_loss: val_rec})
            summary_val_kl = sess.run(merge_op, {value_loss: val_kl})
            writer_val.add_summary(summary_val, i*FLAGS.num_per_val)
            writer_val_rec.add_summary(summary_val_rec, i*FLAGS.num_per_val)
            writer_val_kl.add_summary(summary_val_kl, i*FLAGS.num_per_val)
            val_loss_all.clear()
            val_rec_all.clear()
            val_kl_all.clear()


if __name__ == '__main__':
    main()