from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from model.config import cfg 
from model.ass_fun import *
from net.vtranse_vgg_gan import VTranse

N_cls = cfg.VRD_NUM_CLASS
N_rela = cfg.VRD_NUM_RELA
N_each_batch = cfg.VRD_BATCH_NUM

index_sp = False
index_cls = False

vnet = VTranse()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela, keep_prob_gan = 0.5)

roidb_path = cfg.DIR + 'vtranse/input/vrd_roidb.npz'
res_path = cfg.DIR + 'vtranse/pretrained_para/vrd_vgg_pretrained.ckpt'

roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']
N_train = len(train_roidb)
N_test = len(test_roidb)
N_show = 100
N_save = N_train
N_val = N_test
N_round = 10

lr_init = 0.00001
lambda_cyc = 10

total_var = tf.trainable_variables()
GSO_var = [var for var in total_var if 'gen_GSO' in var.name]
GOS_var = [var for var in total_var if 'gen_GOS' in var.name]
DS_var = [var for var in total_var if 'dis_DS' in var.name]
DO_var = [var for var in total_var if 'dis_DO' in var.name]
encoder_var = [var for var in total_var if 'encoder' in var.name]
res_var = [var for var in total_var if 'vgg_16/conv' in var.name]

GSO_var = GSO_var + encoder_var
GOS_var = GOS_var + encoder_var

saver_res = tf.train.Saver(var_list = res_var)
saver = tf.train.Saver(max_to_keep = 20)

print('res_var:')
for var in res_var:
	print(var)

print('GSO_var:')
for var in GSO_var:
	print(var)

print('GOS_var:')
for var in GOS_var:
	print(var)

print('DS_var:')
for var in DS_var:
	print(var)

print('DO_var:')
for var in DO_var:
	print(var)

optimizer_gen = tf.train.AdamOptimizer(learning_rate = lr_init, beta1=0.5, beta2=0.9)
optimizer_dis = tf.train.GradientDescentOptimizer(learning_rate = lr_init)

train_DO = optimizer_dis.minimize(vnet.gan_losses['o_d_loss'], var_list=DO_var)
train_DS = optimizer_dis.minimize(vnet.gan_losses['s_d_loss'], var_list=DS_var)

train_GOS = optimizer_gen.minimize(vnet.gan_losses['s_g_loss']+lambda_cyc*vnet.gan_losses['L_cyc'], var_list=GOS_var)
train_GSO = optimizer_gen.minimize(vnet.gan_losses['o_g_loss']+lambda_cyc*vnet.gan_losses['L_cyc'], var_list=GSO_var)

train_blob = {}
train_blob['train_DO'] = train_DO
train_blob['train_DS'] = train_DS
train_blob['train_GSO'] = train_GSO
train_blob['train_GOS'] = train_GOS

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	saver_res.restore(sess, res_path)

	t = 0.0
	o_L_d_sum = 0.0
	o_L_g_sum = 0.0
	s_L_g_sum = 0.0
	s_L_d_sum = 0.0
	L_cyc_SO_sum = 0.0
	L_cyc_OS_sum = 0.0
	for r in range(N_round):
		for roidb_id in range(N_train):
			roidb_use = train_roidb[roidb_id]
			if len(roidb_use['rela_gt']) == 0:
				continue
			o_L_d, o_L_g, s_L_g, s_L_d, L_cyc_SO, L_cyc_OS = vnet.train_gan(sess, roidb_use, train_blob)
			o_L_d_sum = o_L_d_sum + o_L_d
			o_L_g_sum = o_L_g_sum + o_L_g
			s_L_g_sum = s_L_g_sum + s_L_g
			s_L_d_sum = s_L_d_sum + s_L_d
			L_cyc_SO_sum = L_cyc_SO_sum + L_cyc_SO
			L_cyc_OS_sum = L_cyc_OS_sum + L_cyc_OS
			t = t + 1.0
			if t % N_show == 0:
				print("t: {0}".format(t))
				print("o_L_d: {0}, o_L_g: {1}".format(o_L_d_sum/N_show, o_L_g_sum/N_show))
				print("s_L_d: {0}, s_L_g: {1}".format(s_L_d_sum/N_show, s_L_g_sum/N_show))
				print("L_cyc_SO: {0}, L_cyc_OS: {1}".format(L_cyc_SO/N_show, L_cyc_OS/N_show))
				o_L_d_sum = 0.0
				o_L_g_sum = 0.0
				s_L_g_sum = 0.0
				s_L_d_sum = 0.0
				L_cyc_SO_sum = 0.0
				L_cyc_OS_sum = 0.0
			if t % N_save == 0: 
				save_path = cfg.DIR + 'vtranse/pred_para/vrd_vgg_gan/vrd_vgg_gan' + format(int(t/N_save),'04') + '.ckpt' 
				print("saving model to {0}".format(save_path))
				saver.save(sess, save_path)



