#data for relationship detection
#need vrd_roidb.npz and vrd_detected_box.npz
import numpy as np 
from model.config import cfg 
from model.ass_fun import *

N_each_batch = cfg.VRD_BATCH_NUM_RELA
N_each_pair = cfg.VRD_AU_PAIR
iou_l = cfg.VRD_IOU_TRAIN
roidb_path = cfg.DIR + 'vtranse/input/vrd_roidb.npz'
detected_box_path = cfg.DIR + 'vtranse/input/vrd_detected_box.npz'
save_path = cfg.DIR + 'vtranse/input/vrd_rela_roidb.npz'

roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']

roidb_read = read_roidb(detected_box_path)
train_detected_box = roidb_read['train_detected_box']
test_detected_box = roidb_read['test_detected_box']

N_train = len(train_roidb)
N_test = len(test_roidb)

roidb = []
for i in range(N_train):
	if (i+1)%100 == 0:
		print(i+1)
	train_roidb_use = train_roidb[i]
	train_detected_box_use = train_detected_box[i]
	roidb_temp = generate_train_rela_roidb(train_roidb_use, train_detected_box_use, iou_l, N_each_batch, N_each_pair)
	roidb.append(roidb_temp)
train_roidb_new = roidb

roidb = []
for i in range(N_test):
	if (i+1)%100 == 0:
		print(i+1)
	test_roidb_use = test_roidb[i]
	test_detected_box_use = test_detected_box[i]
	roidb_temp = generate_train_rela_roidb(test_roidb_use, test_detected_box_use, iou_l, N_each_batch, N_each_pair)
	roidb.append(roidb_temp)
val_roidb_new = roidb

roidb = []
for i in range(N_test):
	if (i+1)%100 == 0:
		print(i+1)
	test_roidb_use = test_roidb[i]
	test_detected_box_use = test_detected_box[i]
	roidb_temp = generate_test_rela_roidb(test_roidb_use, test_detected_box_use, N_each_batch)
	roidb.append(roidb_temp)
test_roidb_new = roidb

roidb = {}
roidb['train_roidb'] = train_roidb_new
roidb['test_roidb'] = test_roidb_new
roidb['val_roidb'] = val_roidb_new

np.savez(save_path, roidb=roidb)