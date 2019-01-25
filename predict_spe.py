import tensorflow as tf
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import csv
import dataIO as io
from network import *
from model import Variational_Autoencoder
import utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():
    # tf flag
    flags = tf.flags
    flags.DEFINE_string("ground_truth_txt", 'F:/data_info/VAE_liver/set_5/PCA/alpha_0.1/fold_1/test_label.txt', "ground truth txt")
    flags.DEFINE_string("indir", 'G:/experiment_result/liver/VAE/set_5/down/64/alpha_0.1/fold_1/VAE/axis_4/beta_6', "input dir")
    flags.DEFINE_string("outdir", 'G:/experiment_result/liver/VAE/set_5/down/64/alpha_0.1/fold_1/VAE/axis_4/beta_6/random', "outdir")
    flags.DEFINE_integer("model_index", 3450 ,"index of model")
    flags.DEFINE_string("gpu_index", "0", "GPU-index")
    flags.DEFINE_float("beta", 1.0, "hyperparameter beta")
    flags.DEFINE_integer("num_of_generate", 1000, "number of generate data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 4, "latent dim")
    flags.DEFINE_list("image_size", [56, 72, 88, 1], "image size")
    FLAGS = flags.FLAGS

    np.random.seed(1)

    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir)

    # load ground truth
    ground_truth = io.load_matrix_data(FLAGS.ground_truth_txt, 'int32')
    print(ground_truth.shape)

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
        mean = np.loadtxt(os.path.join(FLAGS.indir, 'mean_{}.txt'.format(FLAGS.model_index)))
        var = np.loadtxt(os.path.join(FLAGS.indir, 'var_{}.txt'.format(FLAGS.model_index)))
        specificity = []

        tbar = tqdm(range(FLAGS.num_of_generate), ascii=True)
        for i in tbar:
            sample_z = np.random.normal(0, 1.0, (1, FLAGS.latent_dim))
            sample_z = np.asarray(mean) + np.sqrt(np.asarray(var)) * sample_z
            generate_data = VAE.generate_sample(sample_z)
            generate_data = generate_data[0, :, :, :, 0]

            # EUDT
            eudt_image = sitk.GetImageFromArray(generate_data)
            eudt_image.SetSpacing([1, 1, 1])
            eudt_image.SetOrigin([0, 0, 0])

            # label
            label = np.where(generate_data > 0.5, 0, 1)
            label = label.astype(np.int8)
            label_image = sitk.GetImageFromArray(label)
            label_image.SetSpacing([1, 1, 1])
            label_image.SetOrigin([0, 0, 0])

            # # calculate ji
            case_max_ji = 0.
            for image_index in range(ground_truth.shape[0]):
                ji = utils.jaccard(label, ground_truth[image_index])
                if ji > case_max_ji:
                    case_max_ji = ji
            specificity.append([case_max_ji])

            # # output image
            # io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'EUDT', str(i+1))))
            # io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'label', str(i + 1))))

    print('specificity = %f' % np.mean(specificity))

    # # output csv file
    with open(os.path.join(FLAGS.outdir, 'specificity_{}.csv'.format(FLAGS.model_index)), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(specificity)
        writer.writerow(['specificity:', np.mean(specificity)])


if __name__ == '__main__':
    main()