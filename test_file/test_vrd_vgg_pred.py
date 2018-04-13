from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from model.config import cfg 
from model.ass_fun import *
from net.vtranse_vgg import VTranse

N_cls = cfg.VRD_NUM_CLASS
N_rela = cfg.VRD_NUM_RELA
N_each_batch = cfg.VRD_BATCH_NUM

index_sp = False
index_cls = False

vnet = VTranse()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

roidb_path = cfg.DIR + 'vtranse/input/vrd_roidb.npz'
model_path = cfg.DIR + 'vtranse/pred_para/vrd_vgg/vrd_vgg0020.ckpt'
save_path = cfg.DIR + 'vtranse/pred_res/vrd_pred_roidb.npz'

roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']
N_train = len(train_roidb)
N_test = len(test_roidb)

saver = tf.train.Saver()

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	saver.restore(sess, model_path)
	pred_roidb = []
	for roidb_id in range(N_test):
		if (roidb_id+1)%10 == 0:
			print(roidb_id + 1)
		roidb_use = test_roidb[roidb_id]
		if len(roidb_use['rela_gt']) == 0:
			pred_roidb.append({})
			continue
		pred_rela, pred_rela_score = vnet.test_predicate(sess, roidb_use)
		pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,
							'sub_box_dete': roidb_use['sub_box_gt'], 'obj_box_dete': roidb_use['obj_box_gt'],
							'sub_dete': roidb_use['sub_gt'], 'obj_dete': roidb_use['obj_gt']}
		pred_roidb.append(pred_roidb_temp)
roidb = {}
roidb['pred_roidb'] = pred_roidb
np.savez(save_path, roidb=roidb)