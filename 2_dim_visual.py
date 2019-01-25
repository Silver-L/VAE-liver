import tensorflow as tf
import os
import numpy as np
import SimpleITK as sitk
import dataIO as io
from network import *
from model import Variational_Autoencoder
import utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():
    # tf flag
    flags = tf.flags
    flags.DEFINE_string("model", 'G:/experiment_result/liver/VAE/set_4/down_64/RBF/alpha_0.1/4/beta_10/model/model_{}'.format(1350), "model")
    flags.DEFINE_string("outdir", 'G:/experiment_result/liver/VAE/set_4/down_64/RBF/alpha_0.1/4/beta_10/random', "outdir")
    flags.DEFINE_string("gpu_index", "0", "GPU-index")
    flags.DEFINE_float("beta", 1.0, "hyperparameter beta")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 2, "latent dim")
    flags.DEFINE_list("image_size", [56, 72, 88, 1], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir)

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
        VAE.restore_model(FLAGS.model)

        # 2 dim vis
        for j in range(-2, 3):
            for i in range(-2, 3):
                mean = [0.37555057, 0.8882291]
                var = [32.121346, 24.540127]

                sample_z = [[i, j]]
                sample_z = np.asarray(mean) + np.sqrt(np.asarray(var)) * sample_z
                generate_data = VAE.generate_sample(sample_z)
                generate_data = generate_data[0, :, :, :, 0]

                # EUDT
                generate_data = generate_data.astype(np.float32)
                eudt_image = sitk.GetImageFromArray(generate_data)
                eudt_image.SetSpacing([1, 1, 1])
                eudt_image.SetOrigin([0, 0, 0])

                # label
                label = np.where(generate_data > 0.5, 0, 1)
                label = label.astype(np.int16)
                label_image = sitk.GetImageFromArray(label)
                label_image.SetSpacing([1, 1, 1])
                label_image.SetOrigin([0, 0, 0])

                io.write_mhd_and_raw(label_image,
                                     '{}.mhd'.format(os.path.join(FLAGS.outdir, '2_dim', str(i) + '_' + str(j))))


# # load tfrecord function
def _parse_function(record, image_size=[512, 512, 1]):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    image = parsed_features['img_raw']
    image = tf.reshape(image, image_size)
    return image


if __name__ == '__main__':
    main()