from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from model.config import cfg 
from model.ass_fun import *
from net.vtranse_vgg import VTranse

N_cls = cfg.VG_NUM_CLASS
N_rela = cfg.VG_NUM_RELA
N_each_batch = cfg.VG_BATCH_NUM
N_round = cfg.VG_TRAIN_ROUND 
lr_init = cfg.VG_LR_INIT
N_show = 500
N_save = 10000
N_val = 2000

index_sp = False
index_cls = False

vnet = VTranse()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

roidb_path = cfg.DIR + 'vtranse/input/vg_roidb.npz'
res_path = cfg.DIR + 'vtranse/pretrained_para/vg_vgg_pretrained.ckpt'

roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']
N_train = len(train_roidb)
N_test = len(test_roidb)

total_var = tf.trainable_variables()
restore_var = [var for var in total_var if 'vgg_16' in var.name]
cls_score_var = [var for var in total_var if 'cls_score' in var.name]
res_var = [item for item in restore_var if item not in cls_score_var]

saver_res = tf.train.Saver(var_list = restore_var)

RD_var = [var for var in total_var if 'RD' in var.name]

saver = tf.train.Saver(max_to_keep = 200)
for var in RD_var:
	print(var)

optimizer = tf.train.AdamOptimizer(learning_rate=lr_init)
train_loss = vnet.losses['rd_loss']
RD_train = optimizer.minimize(train_loss, var_list = RD_var)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	saver_res.restore(sess, res_path)

	t = 0.0
	rd_loss = 0.0
	acc = 0.0
	for r in range(N_round):
		for roidb_id in range(N_train):
			roidb_use = train_roidb[roidb_id]
			if len(roidb_use['rela_gt']) == 0:
				continue
			rd_loss_temp, acc_temp = vnet.train_predicate(sess, roidb_use, RD_train)
			rd_loss = rd_loss + rd_loss_temp
			acc = acc + acc_temp
			t = t + 1.0
			if t % N_show == 0:
				print("t: {0}, rd_loss: {1}, acc: {2}".format(t, rd_loss/N_show, acc/N_show))
				rd_loss = 0.0
				acc = 0.0
			if t % N_save == 0: 
				save_path = cfg.DIR + 'vtranse/pred_para/vg_vgg/vg_vgg' + format(int(t/N_save),'04') + '.ckpt' 
				print("saving model to {0}".format(save_path))
				saver.save(sess, save_path)
				rd_loss_val = 0.0
				acc_val = 0.0
				for val_id in range(N_val):
					roidb_use = test_roidb[val_id]
					if len(roidb_use['rela_gt']) == 0:
						continue
					rd_loss_temp, acc_temp = vnet.val_predicate(sess, roidb_use)
					rd_loss_val = rd_loss_val + rd_loss_temp
					acc_val = acc_val + acc_temp
				print("val: rd_loss: {0}, acc: {1}".format(rd_loss_val/N_val, acc_val/N_val))








