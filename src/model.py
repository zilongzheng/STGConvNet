from __future__ import division

import tensorflow as tf
import numpy as np
import os
from ops import *
from util import *
from progressbar import ETA, Bar, Percentage, ProgressBar

class STGConvnet(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.net_type = 'FC_S_large'
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.num_frames = config.num_frames
        self.num_chain = config.num_chain
        self.num_epochs = config.num_epochs

        self.lr = config.lr
        self.beta1 = config.beta1
        self.step_size = config.step_size
        self.sample_steps = config.sample_steps

        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.train_dir = os.path.join(self.output_dir, 'observed_sequence')
        self.sample_dir = os.path.join(self.output_dir, 'synthesis_sequence')
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.result_dir = os.path.join(self.output_dir, 'final_result')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        self.syn = tf.placeholder(shape=[self.num_chain, self.num_frames, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.obs = tf.placeholder(shape=[None, self.num_frames, self.image_size, self.image_size, 3], dtype=tf.float32)


    def descriptor(self, inputs, reuse=False):
        with tf.variable_scope('des', reuse=reuse):
            if self.net_type == 'ST':
                """
                This is the spatial temporal model used for synthesizing dynamic textures with both spatial and temporal 
                stationarity. e.g. sea, ocean.
                """
                conv1 = conv3d(inputs, 120, (15, 15, 15), strides=(7, 7, 7), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 40, (7, 7, 7), strides=(3, 3, 3), padding="SAME", name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 20, (2, 3, 3), strides=(1, 2, 2), padding="SAME", name="conv3")
                conv3 = tf.nn.relu(conv3)
                return conv3
            elif self.net_type == 'FC_S':
                """
                This is the spatial fully connected model used for synthesizing dynamic textures with only temporal 
                stationarity with image size of 100. e.g. fire pot, flashing lights.
                """
                conv1 = conv3d(inputs, 120, (7, 7, 7), strides=(2, 2, 2), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 30, (5, 50, 50), strides=(2, 2, 2), padding=(2, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 5, (2, 1, 1), strides=(1, 2, 2), padding=(1, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)

                return conv3
            elif self.net_type == 'FC_S_large':
                """
                This is the spatial fully connected model for images with size of 224.
                """
                conv1 = conv3d(inputs, 120, (7, 7, 7), strides=(3, 3, 3), padding="SAME", name="conv1")
                conv1 = tf.nn.relu(conv1)

                conv2 = conv3d(conv1, 30, (4, 75, 75), strides=(2, 1, 1), padding=(2, 0, 0), name="conv2")
                conv2 = tf.nn.relu(conv2)

                conv3 = conv3d(conv2, 5, (2, 1, 1), strides=(1, 1, 1), padding=(1, 0, 0), name="conv3")
                conv3 = tf.nn.relu(conv3)

                return conv3
            else:
                return NotImplementedError

    def langevin_dynamics(self, samples, gradient, batch_id):
        for i in range(self.sample_steps):
            noise = np.random.randn(*samples.shape)
            grad = self.sess.run(gradient, feed_dict = {self.syn: samples})
            samples = samples - 0.5 * self.step_size * self.step_size * (samples - grad) + self.step_size * noise
            self.pbar.update(batch_id * self.sample_steps + i)
        return samples

    def train(self):

        obs_res = self.descriptor(self.obs, reuse=False)
        syn_res = self.descriptor(self.syn, reuse=True)
        train_loss = tf.subtract(tf.reduce_mean(syn_res,axis=0), tf.reduce_mean(obs_res,axis=0))
        train_loss_mean, train_loss_update = tf.contrib.metrics.streaming_mean(train_loss)

        recon_err_mean, recon_err_update = tf.contrib.metrics.streaming_mean_squared_error(
            tf.reduce_mean(self.syn,axis=0),tf.reduce_mean(self.obs,axis=0))

        dLdI = tf.gradients(syn_res, self.syn)[0]

        # Prepare training data
        loadVideoToFrames(self.data_path, self.train_dir)
        train_data = getTrainingData(self.train_dir, num_frames=self.num_frames, image_size=self.image_size)
        img_mean = train_data.mean()
        train_data = train_data - img_mean
        print(train_data.shape)

        num_batches = int(math.ceil(len(train_data) / self.batch_size))

        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        accum_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in des_vars]
        reset_grads = [var.assign(tf.zeros_like(var)) for var in accum_vars]

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.beta1)
        grads_and_vars = optimizer.compute_gradients(train_loss, var_list=des_vars)
        update_grads = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads_and_vars)]
        des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in grads_and_vars if '/w' in var.name]
        # update by mean of gradients
        apply_grads = optimizer.apply_gradients([(tf.divide(accum_vars[i], num_batches), gv[1]) for i, gv in enumerate(grads_and_vars)])


        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        sample_size = self.num_chain * num_batches
        sample_video = np.random.randn(sample_size, self.num_frames, self.image_size, self.image_size, 3)

        tf.summary.scalar('train_loss', train_loss_mean)
        tf.summary.scalar('recon_err', recon_err_mean)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=50)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        for epoch in range(self.num_epochs):

            gradients = []

            widgets = ["Epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            self.pbar = ProgressBar(maxval=num_batches * self.sample_steps, widgets=widgets)
            self.pbar.start()

            self.sess.run(reset_grads)
            for i in range(num_batches):
                obs_data = train_data[i * self.batch_size:min(len(train_data), (i+1) * self.batch_size)]
                syn = sample_video[i * self.num_chain:(i+1) * self.num_chain]

                syn = self.langevin_dynamics(syn, dLdI, i)

                grad = self.sess.run([des_grads, update_grads, train_loss_update], feed_dict={self.obs: obs_data, self.syn: syn})[0]

                self.sess.run(recon_err_update, feed_dict={self.obs: obs_data, self.syn: syn})

                sample_video[i * self.num_chain:(i + 1) * self.num_chain] = syn

                gradients.append(np.mean(grad))
            self.pbar.finish()

            self.sess.run(apply_grads)
            [loss, recon_err, summary] = self.sess.run([train_loss_mean, recon_err_mean, summary_op])
            print('Epoch #%d, descriptor loss: %.4f, SSD weight: %4.4f, Avg MSE: %4.4f' % (epoch, loss, float(np.mean(gradients)), recon_err))
            writer.add_summary(summary, epoch)

            if epoch % self.log_step == 0:
                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                saveSampleSequence(sample_video + img_mean, self.sample_dir, epoch, col_num=10)

                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(self.sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

            if epoch % 20 == 0:
                saveSampleVideo(sample_video + img_mean, self.result_dir, original=(train_data + img_mean), global_step=epoch)

