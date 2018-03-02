import argparse
import tensorflow as tf
from src.model import STGConvnet

FLAGS = tf.app.flags.FLAGS

# model hyper-parameters
tf.flags.DEFINE_integer('image_size', 224, 'Image size to rescale images')
tf.flags.DEFINE_integer('batch_size', 1, 'Batch size of training images')
tf.flags.DEFINE_integer('num_chain', 3, 'Number of synthesized videos for each batch')
tf.flags.DEFINE_integer('num_epochs', 500, 'Number of epochs to train')
tf.flags.DEFINE_integer('num_frames', 70, 'Number of frames for each training video')

# parameters for descriptorNet
tf.flags.DEFINE_float('lr', 0.001, 'Initial learning rate for descriptorNet')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam')
tf.flags.DEFINE_float('refsig', 1, 'Standard deviation for reference distribution')
tf.flags.DEFINE_integer('sample_steps', 20, 'Sample steps for Langevin dynamics')
tf.flags.DEFINE_float('step_size', 0.3, 'Step size for descriptor Langevin dynamics')

# misc
tf.flags.DEFINE_string('data_dir', './trainingVideo', 'The data directory')
tf.flags.DEFINE_string('category', 'fire_pot', 'The name of dataset')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 10, 'Number of epochs to save output results')
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load')
tf.flags.DEFINE_integer('sample_size', 100, 'Number of images to generate during test.')


def main():
    model = STGConvnet(
        net_type= 'FC_S_large',
        num_epochs=FLAGS.num_epochs, image_size=FLAGS.image_size, num_frames=FLAGS.num_frames,
        batch_size=FLAGS.batch_size, num_chain=FLAGS.num_chain, lr=FLAGS.lr, beta1= FLAGS.beta1,
        refsig=FLAGS.refsig, sample_steps= FLAGS.sample_steps, step_size= FLAGS.step_size,
        data_path=FLAGS.data_dir, category=FLAGS.category, output_dir= FLAGS.output_dir, log_step=FLAGS.log_step
    )

    with tf.Session() as sess:
        model.train(sess)

if __name__ == '__main__':
    main()