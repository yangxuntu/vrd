import numpy as np 
import xlwt
import h5py
import cv2
from model.config import cfg 
from model.ass_fun import *

N_each_pred = cfg.VG_BATCH_NUM2

file_path = cfg.DIR + 'dataset/VG/imdb/vg1_2_meta.h5'
image_path = cfg.DIR + 'dataset/VG/images/VG_100K/'
save_path = cfg.DIR + 'vtranse/input/vg_roidb2.npz'
hdf5_path = ['gt/train/','gt/test/']
f = h5py.File(file_path, "r")

for r in range(2):
	img_list = f[hdf5_path[r]].keys()
	len_img = len(img_list)
	roidb = []

	for image_id in range(len_img):
		if (image_id+1)%1000 == 0:
			print('image id is {0}'.format(image_id+1))
		roidb_temp = {}
		image_path_use = image_path + img_list[image_id] + '.jpg'
		im = cv2.imread(image_path_use)
		if type(im) == type(None):
			continue

		im_shape = np.shape(im)
		im_h = im_shape[0]
		im_w = im_shape[1]

		roidb_temp['image'] = image_path_use
		roidb_temp['width'] = im_w
		roidb_temp['height'] = im_h

		d = f[hdf5_path[r]+img_list[image_id]]
		roidb_temp['sub_box_gt'] = d['sub_boxes'][:] + 0.0
		roidb_temp['obj_box_gt'] = d['obj_boxes'][:] + 0.0
		rlp_labels = d['rlp_labels'][:] + 0.0
		roidb_temp['sub_gt'] = rlp_labels[:,0]
		roidb_temp['obj_gt'] = rlp_labels[:,2]
		roidb_temp['rela_gt'] = rlp_labels[:,1]
		#roidb_temp['index_pred'] = generate_batch(len(rlp_labels), N_each_pred)
		roidb_temp['index_pred'] = generate_batch_bal(rlp_labels, N_each_pred)
		roidb.append(roidb_temp)

	if r == 0:	
		train_roidb = roidb
	elif r == 1:
		test_roidb = roidb
roidb = {}

roidb['train_roidb'] = train_roidb
roidb['test_roidb'] = test_roidb

np.savez(save_path, roidb=roidb)

