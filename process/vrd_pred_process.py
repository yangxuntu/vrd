# data for predicate detection
import numpy as np 
import xlwt
import h5py
import cv2
from model.config import cfg 
from model.ass_fun import *
import json

N_each_pred = cfg.VRD_BATCH_NUM

train_file_path = cfg.DIR + 'dataset/VRD/json_dataset/annotations_train.json'
test_file_path = cfg.DIR + 'dataset/VRD/json_dataset/annotations_test.json'
file_path = [train_file_path, test_file_path]
train_image_path = cfg.DIR + 'dataset/VRD/sg_dataset/sg_train_images/'
test_image_path = cfg.DIR + 'dataset/VRD/sg_dataset/sg_test_images/'
image_path = [train_image_path, test_image_path]
save_path =cfg.DIR + 'vtranse/input/vrd_roidb.npz'

for r in range(2):
	file_path_use = file_path[r]
	image_path_use = image_path[r]
	save_path_use = save_path[r]
	roidb = []
	with open(file_path_use,'r') as f:
		data=json.load(f)
		image_name = data.keys()
		len_img = len(image_name)
		t = 0
		for image_id in range(len_img):
			if (image_id+1)%1000 == 0:
				print('image id is {0}'.format(image_id+1))
			roidb_temp = {}
			image_full_path = image_path_use + image_name[image_id]
			im = cv2.imread(image_full_path)
			if type(im) == type(None):
				continue

			im_shape = np.shape(im)
			im_h = im_shape[0]
			im_w = im_shape[1]

			roidb_temp['image'] = image_full_path
			roidb_temp['width'] = im_w
			roidb_temp['height'] = im_h

			d = data[image_name[image_id]]
			relation_length = len(d)
			if relation_length == 0:
				continue
			sb_new = np.zeros(shape=[relation_length,4])
			ob_new = np.zeros(shape=[relation_length,4])
			rela = np.zeros(shape=[relation_length,])
			obj = np.zeros(shape=[relation_length,])
			subj = np.zeros(shape=[relation_length,])

			for relation_id in range(relation_length):
				relation = d[relation_id]

				obj[relation_id] = relation['object']['category']
				subj[relation_id] = relation['subject']['category']
				rela[relation_id] = relation['predicate']

				ob_temp = relation['object']['bbox']
				sb_temp = relation['subject']['bbox']
				ob = [ob_temp[0],ob_temp[1],ob_temp[2],ob_temp[3]]
				sb = [sb_temp[0],sb_temp[1],sb_temp[2],sb_temp[3]]
				
				ob_new[relation_id][0:4] = [ob[2],ob[0],ob[3],ob[1]]
				sb_new[relation_id][0:4] = [sb[2],sb[0],sb[3],sb[1]]

			roidb_temp['sub_box_gt'] = sb_new[:] + 0.0
			roidb_temp['obj_box_gt'] = ob_new + 0.0
			roidb_temp['sub_gt'] = subj + 0.0
			roidb_temp['obj_gt'] = obj + 0.0
			roidb_temp['rela_gt'] = rela + 0.0
			roidb_temp['index_pred'] = generate_batch(len(rela), N_each_pred)
			roidb.append(roidb_temp)

		if r == 0:	
			train_roidb = roidb
		elif r == 1:
			test_roidb = roidb
roidb = {}

roidb['train_roidb'] = train_roidb
roidb['test_roidb'] = test_roidb

np.savez(save_path, roidb=roidb)

