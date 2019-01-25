import tensorflow as tf
import os
import numpy as np
import csv
from tqdm import tqdm
import SimpleITK as sitk
import dataIO as io
from network import *
from model import Variational_Autoencoder
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("test_data_txt", 'F:/data_info/VAE_liver/set_5/TFrecord/fold_1/test.txt', "test data txt")
    flags.DEFINE_string("indir", 'G:/experiment_result/liver/VAE/set_5/down/64/alpha_0.1/fold_1/VAE/axis_5/beta_7', "input dir")
    flags.DEFINE_string("outdir", 'G:/experiment_result/liver/VAE/set_5/down/64/alpha_0.1/fold_1/VAE/axis_5/beta_7/rec', "outdir")
    flags.DEFINE_integer("model_index", 3300 ,"index of model")
    flags.DEFINE_string("gpu_index", "0", "GPU-index")
    flags.DEFINE_float("beta", 1.0, "hyperparameter beta")
    flags.DEFINE_integer("num_of_test", 75, "number of test data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 5, "latent dim")
    flags.DEFINE_list("image_size", [56, 72, 88, 1], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir)

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

        sess.run(init_op)

        # testing
        VAE.restore_model(os.path.join(FLAGS.indir,'model','model_{}'.format(FLAGS.model_index)))
        tbar = tqdm(range(test_step), ascii=True)
        preds = []
        ori = []
        ji = []
        for k in tbar:
            test_data_batch = sess.run(test_data)
            ori_single = test_data_batch
            preds_single = VAE.reconstruction_image(ori_single)
            preds_single = preds_single[0, :, :, :, 0]
            ori_single = ori_single[0, :, :, :, 0]

            preds.append(preds_single)
            ori.append(ori_single)

            # # label
            ji = []
            for j in range(len(preds)):

                # EUDT
                eudt_image = sitk.GetImageFromArray(preds[j])
                eudt_image.SetSpacing([1, 1, 1])
                eudt_image.SetOrigin([0, 0, 0])

                label = np.where(preds[j] > 0.5, 0, 1)
                # label = np.where(preds[j] > 0.5, 1, 0.5)
                label = label.astype(np.int16)
                label_image = sitk.GetImageFromArray(label)
                label_image.SetSpacing([1, 1, 1])
                label_image.SetOrigin([0, 0, 0])

                ori_label = np.where(ori[j] > 0.5, 0, 1)
                ori_label_image = sitk.GetImageFromArray(ori_label)
                ori_label_image.SetSpacing([1, 1, 1])
                ori_label_image.SetOrigin([0, 0, 0])

                # # calculate ji
                ji.append([utils.jaccard(label, ori_label)])

                # output image
                io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'label', 'recon_{}'.format(j))))

        generalization = np.mean(ji)
        print('generalization = %f' % generalization)

        # # output csv file
        with open(os.path.join(FLAGS.outdir, 'generalization_{}.csv'.format(FLAGS.model_index)), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(ji)
            writer.writerow(['generalization= ', generalization])


if __name__ == '__main__':
    main()