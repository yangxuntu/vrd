from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np
from model.config import cfg 
from model.ass_fun import *

class VTranse(object):
	def __init__(self):
		self.predictions = {}
		self.losses = {}
		self.layers = {}
		self.gan_layers = {}
		self.gan_losses = {}
		self.feat_stride = [16, ]
		self.scope = 'vgg_16'

	def create_graph(self, N_each_batch, index_sp, index_cls, num_classes, num_predicates, keep_prob_gan = None):
		self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
		self.sbox = tf.placeholder(tf.float32, shape=[N_each_batch, 4])
		self.obox = tf.placeholder(tf.float32, shape=[N_each_batch, 4])
		self.sub_sp_info = tf.placeholder(tf.float32, shape=[N_each_batch, 4])
		self.ob_sp_info = tf.placeholder(tf.float32, shape=[N_each_batch, 4])
		self.rela_label = tf.placeholder(tf.int32, shape=[N_each_batch,])
		self.keep_prob = tf.placeholder(tf.float32)
		self.keep_prob_gan = keep_prob_gan
		self.index_sp = index_sp
		self.index_cls = index_cls
		self.num_classes = num_classes
		self.num_predicates = num_predicates
		self.N_each_batch = N_each_batch

		self.build_dete_network()
		self.build_gan_network()
		self.build_rd_network3()
		self.add_rd_loss()


	def build_dete_network(self, is_training=True):
		net_conv = self.image_to_head(is_training)
		sub_pool5, sub_fc = self.crop_pool_layer(net_conv, self.sbox, "sub_pool5")
		ob_pool5, ob_fc = self.crop_pool_layer(net_conv, self.obox, "ob_pool5")

		self.layers['sub_pool5'] = sub_pool5
		self.layers['ob_pool5'] = ob_pool5
		self.layers['sub_fc'] = sub_fc
		self.layers['ob_fc'] = ob_fc

	def build_gan_network(self, is_training = True):
		sub_fc = self.layers['sub_fc']
		ob_fc = self.layers['ob_fc']

		self.gan_layers['s_real'] = sub_fc
		self.gan_layers['o_real'] = ob_fc

		self.gan_layers['o_fake'] = self.map_fun_res(self.gan_layers['s_real'], 'gen_GSO_gan', reuse = False)
		self.gan_layers['s_fake'] = self.map_fun_res(self.gan_layers['o_real'], 'gen_GOS_gan', reuse = False)

		self.gan_layers['s_return'] = self.map_fun_res(self.gan_layers['o_fake'], 'gen_GOS_gan', reuse = True)
		self.gan_layers['o_return'] = self.map_fun_res(self.gan_layers['s_fake'], 'gen_GSO_gan', reuse = True)

		self.gan_losses['L_cyc'] = \
			tf.reduce_mean(tf.abs(self.gan_layers['s_real'] - self.gan_layers['s_return'])
			 + tf.abs(self.gan_layers['o_real'] - self.gan_layers['o_return']))

		self.gan_layers['s_dis_real'] = \
			self.dis_gan(self.gan_layers['s_real'], 'dis_DS_gan', self.keep_prob_gan, reuse = False)
		self.gan_layers['s_dis_fake'] = \
			self.dis_gan(self.gan_layers['s_fake'], 'dis_DS_gan', self.keep_prob_gan, reuse = True)

		output_shape = tf.shape(self.gan_layers['s_dis_real'])
		real_label1 = tf.random_uniform(output_shape, 0.8, 1.2)
		fake_label = tf.random_uniform(output_shape, 0.0, 0.2)
		self.gan_losses['s_d_loss'] = \
			tf.reduce_mean(tf.square(self.gan_layers['s_dis_real'] - real_label1) 
				+ tf.square(self.gan_layers['s_dis_fake'] - fake_label))
		
		real_label2 = tf.random_uniform(output_shape, 0.8, 1.2)
		self.gan_losses['s_g_loss'] = \
			tf.reduce_mean( tf.square(self.gan_layers['s_dis_fake'] - real_label2))

		self.gan_layers['o_dis_real'] = \
			self.dis_gan(self.gan_layers['o_real'], 'dis_DO_gan', self.keep_prob_gan, reuse = False)
		self.gan_layers['o_dis_fake'] = \
			self.dis_gan(self.gan_layers['o_fake'], 'dis_DO_gan', self.keep_prob_gan, reuse = True)

		output_shape = tf.shape(self.gan_layers['o_dis_real'])
		real_label1 = tf.random_uniform(output_shape, 0.8, 1.2)
		fake_label = tf.random_uniform(output_shape, 0.0, 0.2)
		self.gan_losses['o_d_loss'] = \
			tf.reduce_mean(tf.square(self.gan_layers['o_dis_real'] - real_label1) 
				+ tf.square(self.gan_layers['o_dis_fake'] - fake_label))
		
		real_label2 = tf.random_uniform(output_shape, 0.8, 1.2)
		self.gan_losses['o_g_loss'] = \
			tf.reduce_mean( tf.square(self.gan_layers['o_dis_fake'] - real_label2))

	def map_fun_res(slef, x, name, reuse = False):
		K = int(x.get_shape()[1])
		with tf.variable_scope(name, reuse=reuse) as scope:
			fc1 = fc(x, K, name = "fc1", relu = False, reuse = reuse)
			fc1 = leaky_relu(fc1, 0.2)
			fc1_output = fc1 + x

			fc2 = fc(fc1_output, K, name = "fc2", relu = False, reuse = reuse)
			fc2 = leaky_relu(fc2, 0.2)
			fc2_output = fc2 + fc1_output
		return fc2_output


	def dis_gan(self, x, name, keep_prob, reuse=False):
		with tf.variable_scope(name, reuse=reuse) as scope:
			fc1 = fc(x, 4096, name = 'fc1', relu = False, reuse = reuse)
			fc1 = leaky_relu(fc1, 0.2)
			fc1 = dropout(fc1, keep_prob)

			fc2 = fc(fc1, 2048, name = 'fc2', relu = False, reuse = reuse )
			fc2 = leaky_relu(fc2, 0.2)

			fc2 = dropout(fc2, keep_prob)
			output = fc(fc2, 1, name = 'output', relu = False, reuse = reuse)
			return output

	def image_to_head(self, is_training, reuse=False):
		with tf.variable_scope(self.scope, self.scope, reuse=reuse):
			net = slim.repeat(self.image, 2, slim.conv2d, 64, [3, 3], 
				trainable=is_training, scope='conv1')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
				trainable=is_training, scope='conv2')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
				trainable=is_training, scope='conv3')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
				trainable=is_training, scope='conv4')
			net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
			net_conv = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
				trainable=is_training, scope='conv5')
			
			en1 = conv(net_conv, 1, 1, 128, 1, 1, 'RD_encoder1', relu = False)
			en1 = leaky_relu(en1, 0.2)

			self.layers['head'] = en1
			return en1

	def head_to_tail(self, pool5, is_training, reuse=False):
		with tf.variable_scope(self.scope, self.scope, reuse=reuse):
			pool5_flat = slim.flatten(pool5, scope='flatten')
			fc6 = fc(pool5_flat, 4096, name = "fc6", relu = False, reuse = reuse)
			fc6 = leaky_relu(fc6, 0.2)
			#fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
			#fc6 = slim.dropout(fc6, keep_prob=self.keep_prob, is_training=True, scope='dropout6')
			fc7 = fc(fc6, 4096, name = "fc7", relu = False, reuse = reuse)
			fc7 = leaky_relu(fc7, 0.2)
			#fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
			#fc7 = slim.dropout(fc7, keep_prob=self.keep_prob, is_training=True, scope='dropout7')

			return fc7

	def crop_pool_layer(self, bottom, rois, name):
		"""
		Notice that the input rois is a N*4 matrix, and the coordinates of x,y should be original x,y times im_scale. 
		"""
		with tf.variable_scope(name) as scope:
			n=tf.to_int32(rois.shape[0])
			batch_ids = tf.zeros([n,],dtype=tf.int32)
			# Get the normalized coordinates of bboxes
			bottom_shape = tf.shape(bottom)
			height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.feat_stride[0])
			width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.feat_stride[0])
			x1 = tf.slice(rois, [0, 0], [-1, 1], name="x1") / width
			y1 = tf.slice(rois, [0, 1], [-1, 1], name="y1") / height
			x2 = tf.slice(rois, [0, 2], [-1, 1], name="x2") / width
			y2 = tf.slice(rois, [0, 3], [-1, 1], name="y2") / height
			# Won't be back-propagated to rois anyway, but to save time
			bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
			crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE*2, cfg.POOLING_SIZE*2],
											 name="crops")
			pooling = max_pool(crops, 2, 2, 2, 2, name="max_pooling")
			flat = tf.reshape(pooling, shape=[n, -1])
		return pooling, flat


	def region_classification(self, fc7, is_training, reuse = False):
		cls_score = slim.fully_connected(fc7, self.num_classes, 
										 activation_fn=None, scope='cls_score', reuse=reuse)
		print("cls_score's shape: {0}".format(cls_score.get_shape()))
		cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
		cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

		return cls_prob, cls_pred

	def build_rd_network3(self):
		sub_fc = self.layers['sub_fc']
		ob_fc = self.layers['ob_fc']

		fc_con = tf.concat( [sub_fc, ob_fc], axis = 1)
		fc1 = slim.fully_connected(fc_con, 2048, 
										 activation_fn=tf.nn.relu, scope='RD_fc1')
		fc1 = dropout(fc1, self.keep_prob)

		fc2 = slim.fully_connected(fc1, 1024, 
										 activation_fn=tf.nn.relu, scope='RD_fc2')
		fc2 = dropout(fc2, self.keep_prob)

		rela_score = slim.fully_connected(fc2, self.num_predicates,
										 activation_fn=None, scope='RD_fc3')

		rela_prob = tf.nn.softmax(rela_score)
		self.layers['rela_score'] = rela_score
		self.layers['rela_prob'] = rela_prob

	def add_rd_loss(self):
		rela_score = self.layers['rela_score']
		rela_prob = self.layers['rela_prob']
		rela_label = self.rela_label
		rd_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(
									labels = rela_label, logits = rela_score) )
		self.losses['rd_loss'] = rd_loss

		acc_each = tf.nn.in_top_k(rela_score, rela_label, 1)
		self.losses['acc_each'] = acc_each
		self.losses['acc'] = tf.reduce_mean( tf.cast(acc_each, tf.float32) )

		rela_pred = tf.argmax(rela_score, 1)
		self.predictions['rela_pred'] = rela_pred

		rela_max_prob = tf.reduce_max(rela_prob, 1)
		self.predictions['rela_max_prob'] = rela_max_prob
		self.predictions['rela_prob'] = rela_prob


	def train_predicate(self, sess, roidb_use, RD_train):
		im, im_scale = im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_pred'])/self.N_each_batch
		RD_loss = 0.0
		acc = 0.0
		for batch_id in range(np.int32(batch_num)):
			blob = get_blob_pred(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.rela_label: blob['rela'],
						 self.keep_prob: 0.5}
			_, losses, predictions, layers = sess.run([RD_train, self.losses, self.predictions, self.layers], feed_dict = feed_dict)
			RD_loss = RD_loss + losses['rd_loss']
			acc = acc + losses['acc']
			#print(predictions['rela_prob'])

		RD_loss = RD_loss/batch_num
		acc = acc/batch_num
		return RD_loss, acc

	def train_gan(self, sess, roidb_use, train_blob):
		im, im_scale = im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_pred'])/self.N_each_batch
		train_GSO = train_blob['train_GSO']
		train_GOS = train_blob['train_GOS']
		train_DS = train_blob['train_DS']
		train_DO = train_blob['train_DO']

		o_L_d_sum = 0.0
		o_L_g_sum = 0.0
		s_L_g_sum = 0.0
		s_L_d_sum = 0.0
		L_cyc_SO_sum = 0.0
		L_cyc_OS_sum = 0.0
		for batch_id in range(np.int32(batch_num)):
			blob = get_blob_pred(roidb_use, im_scale, 0, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.keep_prob: 0.5}
			_, o_L_g, L_cyc_SO = \
				sess.run([train_GSO, self.gan_losses['o_g_loss'], self.gan_losses['L_cyc']], feed_dict = feed_dict)
			_, o_L_d = \
				sess.run([train_DO, self.gan_losses['o_d_loss']], feed_dict = feed_dict)

			_, s_L_g, L_cyc_OS = \
				sess.run([train_GOS, self.gan_losses['s_g_loss'], self.gan_losses['L_cyc']], feed_dict = feed_dict)
			_, s_L_d = \
				sess.run([train_DS, self.gan_losses['s_d_loss']], feed_dict = feed_dict)

			o_L_d_sum = o_L_d_sum + o_L_d
			o_L_g_sum = o_L_g_sum + o_L_g
			s_L_g_sum = s_L_g_sum + s_L_g
			s_L_d_sum = s_L_d_sum + s_L_d
			L_cyc_SO_sum = L_cyc_SO_sum + L_cyc_SO
			L_cyc_OS_sum = L_cyc_OS_sum + L_cyc_OS

		o_L_d_sum = o_L_d_sum/batch_num
		o_L_g_sum = o_L_g_sum/batch_num
		s_L_g_sum = s_L_g_sum/batch_num
		s_L_d_sum = s_L_d_sum/batch_num
		L_cyc_SO_sum = L_cyc_SO_sum/batch_num
		L_cyc_OS_sum = L_cyc_OS_sum/batch_num
		return o_L_d_sum, o_L_g_sum, s_L_g_sum, s_L_d_sum, L_cyc_SO_sum, L_cyc_OS_sum

	def train_rela(self, sess, roidb_use, RD_train):
		im, im_scale = im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_rela'])/self.N_each_batch
		RD_loss = 0.0
		acc = 0.0
		for batch_id in range(np.int32(batch_num)):
			blob = get_blob_rela(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.rela_label: blob['rela'],
						 self.keep_prob: 0.5}
			_, losses = sess.run([RD_train, self.losses], feed_dict = feed_dict)
			RD_loss = RD_loss + losses['rd_loss']
			acc = acc + losses['acc']

		RD_loss = RD_loss/batch_num
		acc = acc/batch_num
		return RD_loss, acc

	def val_predicate(self, sess, roidb_use):
		im, im_scale = im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_pred'])/self.N_each_batch
		RD_loss = 0.0
		acc = 0.0
		for batch_id in range(np.int32(batch_num)):
			blob = get_blob_pred(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.rela_label: blob['rela'],
						 self.keep_prob: 1}
			losses = sess.run(self.losses, feed_dict = feed_dict)
			RD_loss = RD_loss + losses['rd_loss']
			acc = acc + losses['acc']

		RD_loss = RD_loss/batch_num
		acc = acc/batch_num
		return RD_loss, acc

	def val_rela(self, sess, roidb_use):
		im, im_scale = im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_rela'])/self.N_each_batch
		RD_loss = 0.0
		acc = 0.0
		for batch_id in range(np.int32(batch_num)):
			blob = get_blob_rela(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.rela_label: blob['rela'],
						 self.keep_prob: 1}
			losses = sess.run(self.losses, feed_dict = feed_dict)
			RD_loss = RD_loss + losses['rd_loss']
			acc = acc + losses['acc']

		RD_loss = RD_loss/batch_num
		acc = acc/batch_num
		return RD_loss, acc

	def test_predicate(self, sess, roidb_use):
		im, im_scale = im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_pred'])/self.N_each_batch
		pred_rela = np.zeros([len(roidb_use['index_pred']),])
		pred_rela_score = np.zeros([len(roidb_use['index_pred']), self.num_predicates])

		for batch_id in range(np.int32(batch_num)):
			blob = get_blob_pred(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.rela_label: blob['rela'],
						 self.keep_prob: 1}
			predictions = sess.run(self.predictions, feed_dict = feed_dict)
			pred_rela[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['rela_pred'][:]
			pred_rela_score[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['rela_prob'][:]
		N_rela = len(roidb_use['rela_gt'])
		pred_rela = pred_rela[0:N_rela]
		pred_rela_score = pred_rela_score[0:rela]
		return pred_rela, pred_rela_score

	def test_rela(self, sess, roidb_use):
		im, im_scale = im_preprocess(roidb_use['image'])
		batch_num = len(roidb_use['index_rela'])/self.N_each_batch
		pred_rela = np.zeros([len(roidb_use['index_rela']),])
		pred_rela_score = np.zeros([len(roidb_use['index_rela']),])

		for batch_id in range(np.int32(batch_num)):
			blob = get_blob_rela(roidb_use, im_scale, self.index_sp, self.N_each_batch, batch_id)
			feed_dict = {self.image: im, self.sbox: blob['sub_box'], self.obox: blob['obj_box'], self.rela_label: blob['rela'],
						 self.keep_prob: 1}
			predictions = sess.run(self.predictions, feed_dict = feed_dict)
			pred_rela[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['rela_pred'][:]
			pred_rela_score[batch_id*self.N_each_batch:(batch_id+1)*self.N_each_batch] = predictions['rela_max_prob'][:]
		N_rela = len(roidb_use['rela_dete'])
		pred_rela = pred_rela[0:N_rela]
		pred_rela_score = pred_rela_score[0:N_rela]
		return pred_rela, pred_rela_score

def conv(x, h, w, K, s_y, s_x, name, relu = True, reuse=False, padding='SAME'):
	"""
	Args:
		x: input
		h: height of filter
		w: width of filter
		K: number of filters
		s_y: stride of height of filter
		s_x: stride of width of filter
	"""
	#c means the number of input channels
	c = int(x.get_shape()[-1])

	with tf.variable_scope(name, reuse=reuse) as scope:
		weights = tf.get_variable('weights', shape=[h,w,c,K])
		biases = tf.get_variable('biases', shape=[K])

		conv_value = tf.nn.conv2d(x, weights, strides = [1,s_y,s_x,1], padding = padding)
		add_baises_value = tf.reshape(tf.nn.bias_add(conv_value, biases), tf.shape(conv_value))
		if relu==True:
			relu_value = tf.nn.relu(add_baises_value, name=scope.name)
		else:
			relu_value = add_baises_value

		return relu_value

def fc(x,K,name,relu=True,reuse=False):
	"""
	Args:
		x: input
		K: the dimension of the output
	"""
	#c means the number of input channels

	c = int(x.get_shape()[1])
	with tf.variable_scope(name, reuse=reuse) as scope:
		weights = tf.get_variable('weights', shape=[c,K])
		biases = tf.get_variable('biases',shape=[K])
		relu_value = tf.nn.xw_plus_b(x,weights,biases,name = scope.name)
		if relu:
			result_value = tf.nn.relu(relu_value)
		else:
			result_value = relu_value
		return result_value

def max_pool(x, h, w, s_y, s_x, name, padding='SAME'):
	return tf.nn.max_pool(x, ksize=[1,h,w,1], strides=[1, s_x, s_y, 1], padding=padding, name=name)

def avg_pool(x, h, w, s_y, s_x, name, padding='SAME'):
	return tf.nn.avg_pool(x, ksize=[1,h,w,1], strides=[1, s_x, s_y, 1], padding=padding, name=name)

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

def leaky_relu(x, alpha):
	return tf.maximum(x, alpha * x)