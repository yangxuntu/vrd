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
N_each_batch = cfg.VRD_BATCH_NUM_RELA

index_sp = False
index_cls = False

vnet = VTranse()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

name = 'pred_res/vrd_rela_res.txt'
for r in range(3,16):
	roidb_path = cfg.DIR + 'vtranse/input/vrd_rela_roidb.npz'
	model_path = cfg.DIR + 'vtranse/pred_para/vrd_vgg_rela/vrd_vgg2' + format(int(r),'04') + '.ckpt'
	save_path = cfg.DIR + 'vtranse/pred_res/vrd_rela_roidb.npz'

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
			pred_rela, pred_rela_score = vnet.test_rela(sess, roidb_use)
			sub_score = roidb_use['sub_score']
			obj_score = roidb_use['obj_score']
			for ii in range(len(pred_rela_score)):
				pred_rela_score[ii] = pred_rela_score[ii]*sub_score[ii]*obj_score[ii]
			pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,
								'sub_box_dete': roidb_use['sub_box_dete'], 'obj_box_dete': roidb_use['obj_box_dete'],
								'sub_dete': roidb_use['sub_dete']-1, 'obj_dete': roidb_use['obj_dete']-1}
			pred_roidb.append(pred_roidb_temp)
	roidb = {}
	roidb['pred_roidb'] = pred_roidb
	np.savez(save_path, roidb=roidb)

	gt_roidb_path = cfg.DIR + 'vtranse/input/vrd_rela_roidb.npz'
	pred_roidb_path = cfg.DIR + 'vtranse/pred_res/vrd_rela_roidb.npz'

	roidb_read = read_roidb(gt_roidb_path)
	train_roidb = roidb_read['train_roidb']
	test_roidb = roidb_read['test_roidb']

	roidb_read = read_roidb(pred_roidb_path)
	pred_roidb = roidb_read['pred_roidb']

	rela_R50, rela_num_right50 = rela_recall(test_roidb, pred_roidb, 50)
	rela_R100, rela_num_right100 = rela_recall(test_roidb, pred_roidb, 100)

	phrase_R50, phrase_num_right50 = phrase_recall(test_roidb, pred_roidb, 50)
	phrase_R100, phrase_num_right100 = phrase_recall(test_roidb, pred_roidb, 100)

	print('rela_R50: {0}, rela_R100: {1}'.format(rela_R50, rela_R100))
	print('phrase_R50: {0}, phrase_R100: {1}'.format(phrase_R50, phrase_R100))

	text_file = open(name,"aw")
	text_file.write('{0}\n'.format('vrd_vgg2' + format(int(r),'04') + '.ckpt'))
	text_file.write('rela_R50: {0}, rela_R100: {1}\n'.format(rela_R50, rela_R100))
	text_file.write('phrase_R50: {0}, phrase_R100: {1}\n'.format(phrase_R50, phrase_R100))
	text_file.close()
