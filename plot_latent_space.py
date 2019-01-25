import tensorflow as tf
import os
import numpy as np
import pickle
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from tqdm import tqdm
import dataIO as io
from network import *
from model import Variational_Autoencoder
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():
    # tf flag
    flags = tf.flags
    flags.DEFINE_string("test_data_txt", 'F:/data_info/VAE_liver/set_5/TFrecord/fold_1/train.txt', "test data txt")
    flags.DEFINE_string("dir", 'G:/experiment_result/liver/VAE/set_5/down/64/alpha_0.1/fold_1/VAE/axis_4/beta_6', "input dir")
    flags.DEFINE_integer("model_index", 3450 ,"index of model")
    flags.DEFINE_string("gpu_index", "0", "GPU-index")
    flags.DEFINE_float("beta", 1.0, "hyperparameter beta")
    flags.DEFINE_integer("num_of_test", 4681, "number of test data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 4, "latent dim")
    flags.DEFINE_list("image_size", [56, 72, 88, 1], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(FLAGS.dir)):
        os.makedirs(FLAGS.dir)

    # read list
    test_data_list = io.load_list(FLAGS.test_data_txt)

    # test step
    test_step = FLAGS.num_of_test // FLAGS.batch_size
    if FLAGS.num_of_test % FLAGS.batch_size != 0:
        test_step += 1

    # load test data
    test_set = tf.data.TFRecordDataset(test_data_list, compression_type = 'GZIP')
    test_set = test_set.map(lambda x: utils._parse_function(x, image_size=FLAGS.image_size),
                            num_parallel_calls=os.cpu_count())
    test_set = test_set.batch(FLAGS.batch_size)
    test_iter = test_set.make_one_shot_iterator()
    test_data = test_iter.get_next()

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config(index=FLAGS.gpu_index)) as sess:

        # set network
        kwargs = {
            'sess': sess,
            'outdir': FLAGS.dir,
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

        sess.run(init_op)

        # testing
        VAE.restore_model(os.path.join(FLAGS.dir,'model','model_{}'.format(FLAGS.model_index)))
        tbar = tqdm(range(test_step), ascii=True)
        latent_space = []
        for k in tbar:
            test_data_batch = sess.run(test_data)
            ori_single = test_data_batch
            z = VAE.plot_latent(ori_single)
            z = z.flatten()
            if FLAGS.latent_dim == 1:
                z = [z[0], 0]
            latent_space.append(z)

        latent_space = np.asarray(latent_space)
        plt.figure(figsize=(8, 6))
        fig = plt.scatter(latent_space[:, 0], latent_space[:, 1], alpha=0.2)
        plt.title('latent distribution')
        plt.xlabel('dim_1')
        plt.ylabel('dim_2')
        plt.savefig(os.path.join(FLAGS.dir, 'latent_distribution_{}.PNG'.format(FLAGS.model_index)))
        # filename = open(os.path.join(FLAGS.outdir, 'latent_distribution.pickle'), 'wb')
        # pickle.dump(fig, filename)
        # plt.show()

        latent_space = np.asarray(latent_space)
        mean = np.average(latent_space, axis=0)
        var = np.var(latent_space, axis=0, ddof=1)
        print(mean)
        print(var)
        print(np.cov(latent_space.transpose()))
        print('skew, kurtosis')
        print(skew(latent_space, axis=0))
        print(kurtosis(latent_space, axis=0))

        # output mean and var
        np.savetxt(os.path.join(FLAGS.dir, 'mean_{}.txt'.format(FLAGS.model_index)), mean)
        np.savetxt(os.path.join(FLAGS.dir, 'var_{}.txt'.format(FLAGS.model_index)), var)


if __name__ == '__main__':
    main()