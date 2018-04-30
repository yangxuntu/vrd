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
N_each_batch = cfg.VG_BATCH_NUM2

index_sp = False
index_cls = False

vnet = VTranse()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

roidb_path = cfg.DIR + 'vtranse/input/vg_roidb.npz'
model_path = cfg.DIR + 'vtranse/pred_para/vg_vgg/vg_vgg0001.ckpt'
save_path = cfg.DIR + 'vtranse/pred_res/vg_pred_roidb.npz'

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
		if (roidb_id + 1)%100 == 0:
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

gt_roidb_path = cfg.DIR + 'vtranse/input/vg_roidb.npz'
pred_roidb_path = cfg.DIR + 'vtranse/pred_res/vg_pred_roidb.npz'

roidb_read = read_roidb(gt_roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']

roidb_read = read_roidb(pred_roidb_path)
pred_roidb = roidb_read['pred_roidb']

R50, num_right50 = rela_recall(test_roidb, pred_roidb, 50)
R100, num_right100 = rela_recall(test_roidb, pred_roidb, 100)

print('R50: {0}, R100: {1}'.format(R50, R100))

