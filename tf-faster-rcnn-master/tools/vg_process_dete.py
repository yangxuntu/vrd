import numpy as np 
import xlwt
import h5py
import cv2

N_obj = 201
file_path = '/home/yangxu/project/rd/dataset/VG/imdb/vg1_2_meta.h5'
image_path = '/home/yangxu/project/rd/dataset/VG/images/VG_100K/'
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
		sub_box = d['sub_boxes'][:] + 0.0
		obj_box = d['obj_boxes'][:] + 0.0
		rlp_labels = d['rlp_labels'][:] + 0.0
		sb = rlp_labels[:,0]
		ob = rlp_labels[:,2]

		boxes = np.concatenate( (sub_box, obj_box), axis=0 )
		unique_boxes, unique_inds = np.unique(boxes, axis=0, return_index = True)

		boxes_label = np.concatenate( (sb,ob), axis=0)
		unique_boxes_label = boxes_label[unique_inds]

		roidb_temp['boxes'] = unique_boxes
		roidb_temp['gt_classes'] = unique_boxes_label
		roidb_temp['max_overlaps'] = np.ones(np.shape(unique_boxes_label))
		roidb_temp['flipped'] = False
		roidb.append(roidb_temp)

	if r == 0:	
		N_roidb = len(roidb)
		for image_id in range(N_roidb):
			roidb_temp = {}
			roidb_temp['image'] = roidb[image_id]['image']
			roidb_temp['width'] = roidb[image_id]['width']
			roidb_temp['height'] = roidb[image_id]['height']
			roidb_temp['gt_classes'] = roidb[image_id]['gt_classes']
			roidb_temp['max_overlaps'] = roidb[image_id]['max_overlaps']
			roidb_temp['flipped'] = True

			boxes_old = roidb[image_id]['boxes']
			width = roidb[image_id]['width']
			boxes_new = np.zeros(np.shape(boxes_old))
			boxes_new[:,0] = width - boxes_old[:,2] - 1
			boxes_new[:,1] = boxes_old[:,1]
			boxes_new[:,2] = width - boxes_old[:,0] - 1
			boxes_new[:,3] = boxes_old[:,3]
			roidb_temp['boxes'] = boxes_new
			roidb.append(roidb_temp)
		train_roidb = roidb
	elif r == 1:
		test_roidb = roidb
roidb = {}

roidb['train_roidb'] = train_roidb
roidb['test_roidb'] = test_roidb

np.savez('vg_roidb.npz',roidb=roidb)

