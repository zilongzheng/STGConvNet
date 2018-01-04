import math
import numpy as numpy
import tensorflow as tf

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		return conv

def conv3d(input_, output_dim, kernal=(5, 5, 5), strides=(2, 2, 2), padding='SAME', stddev=0.02, name="conv3d"):
	if type(kernal) == list or type(kernal) == tuple:
		[k_d, k_h, k_w] = list(kernal)
	else:
		k_d = k_h = k_w = kernal
	if type(strides) == list or type(strides) == tuple:
		[d_d, d_h, d_w] = list(strides)
	else:
		d_d = d_h = d_w = strides

	with tf.variable_scope(name):

		if type(padding) == list or type(padding) == tuple:
			padding = [0] + list(padding) + [0]
			input_ = tf.pad(input_, [[p, p] for p in padding], "CONSTANT")
			padding = 'VALID'
		w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding=padding)
		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.nn.bias_add(conv, biases)
		return conv
