from __future__ import division

import tensorflow as tf
import numpy as np
import os
from ops import *
from util import *
from progressbar import ETA, Bar, Percentage, ProgressBar


class STGConvnet(object):
    def __init__(self, net_type='FC_S_large', batch_size=1, image_size=224, num_frames=70, num_chain=3, num_epochs=1000,
                 lr=0.01, beta1=0.5, refsig=0.1, step_size=0.3, sample_steps=20,
                 data_path='/tmp/data', category='fire_pot', log_step=10, output_dir='./output'):
        self.net_type = net_type
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_chain = num_chain
        self.num_epochs = num_epochs

        self.lr = lr
        self.beta1 = beta1
        self.step_size = step_size
        self.refsig = refsig
        self.sample_steps = sample_steps

        self.data_path = os.path.join(data_path, category)
        self.log_step = log_step
        self.output_dir = os.path.join(output_dir, category)

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.train_dir = os.path.join(self.output_dir, 'observed_sequence')
        self.sample_dir = os.path.join(self.output_dir, 'synthesis_sequence')
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.result_dir = os.path.join(self.output_dir, 'final_result')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        self.syn = tf.placeholder(shape=[self.num_chain, self.num_frames, self.image_size, self.image_size, 3],
                                  dtype=tf.float32)
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

    def langevin_dynamics(self, syn_arg):
        def cond(i, syn):
            return tf.less(i, self.sample_steps)

        def body(i, syn):
            noise = tf.random_normal(shape=tf.shape(syn), name='noise')
            syn_res = self.descriptor(syn, reuse=True)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            syn = syn - 0.5 * self.step_size * self.step_size * (syn / self.refsig / self.refsig - grad)
            syn = syn + self.step_size * noise
            return tf.add(i, 1), syn

        with tf.name_scope("langevin_dynamics"):
            i = tf.constant(0)
            i, syn = tf.while_loop(cond, body, [i, syn_arg])
            return syn

    def train(self, sess):

        obs_res = self.descriptor(self.obs, reuse=False)
        syn_res = self.descriptor(self.syn, reuse=True)
        train_loss = tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0))
        train_loss_mean, train_loss_update = tf.contrib.metrics.streaming_mean(train_loss)

        recon_err_mean, recon_err_update = tf.contrib.metrics.streaming_mean_squared_error(
            tf.reduce_mean(self.syn, axis=0), tf.reduce_mean(self.obs, axis=0))

        langevin_descriptor = self.langevin_dynamics(self.syn)

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
        apply_grads = optimizer.apply_gradients(
            [(tf.divide(accum_vars[i], num_batches), gv[1]) for i, gv in enumerate(grads_and_vars)])

        tf.summary.scalar('train_loss', train_loss_mean)
        tf.summary.scalar('recon_err', recon_err_mean)
        summary_op = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        sample_size = self.num_chain * num_batches
        sample_video = np.random.randn(sample_size, self.num_frames, self.image_size, self.image_size, 3)

        saver = tf.train.Saver(max_to_keep=50)
        writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # make graph immutable
        tf.get_default_graph().finalize()

        # store graph in protobuf
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        with open(self.model_dir + '/graph.proto', 'w') as f:
            f.write(str(tf.get_default_graph().as_graph_def()))

        for epoch in xrange(self.num_epochs):

            gradients = []

            sess.run(reset_grads)
            for i in xrange(num_batches):
                obs_data = train_data[i * self.batch_size:min(len(train_data), (i + 1) * self.batch_size)]
                syn = sample_video[i * self.num_chain:(i + 1) * self.num_chain]

                syn = sess.run(langevin_descriptor, feed_dict={self.syn: syn})

                grad = sess.run([des_grads, update_grads, train_loss_update],
                                     feed_dict={self.obs: obs_data, self.syn: syn})[0]

                sess.run(recon_err_update, feed_dict={self.obs: obs_data, self.syn: syn})

                sample_video[i * self.num_chain:(i + 1) * self.num_chain] = syn

                gradients.append(grad)

            sess.run(apply_grads)
            [loss, recon_err, summary] = sess.run([train_loss_mean, recon_err_mean, summary_op])
            print('Epoch #%d, descriptor loss: %.4f, SSD weight: %4.4f, Avg MSE: %4.4f' % (
            epoch, loss, float(np.mean(gradients)), recon_err))
            writer.add_summary(summary, epoch)

            if epoch % self.log_step == 0:
                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                saveSampleSequence(sample_video + img_mean, self.sample_dir, epoch, col_num=10)

                saver.save(sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

            if epoch % 20 == 0:
                saveSampleVideo(sample_video + img_mean, self.result_dir, original=(train_data + img_mean),
                                global_step=epoch)
