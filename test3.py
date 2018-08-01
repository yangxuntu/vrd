from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
from model.config import cfg 
from model.ass_fun import *
from net.vtranse_vgg import VTranse
import cv2
import matplotlib.pyplot as plt

N_cls = cfg.VRD_NUM_CLASS
N_rela = cfg.VRD_NUM_RELA
N_each_batch = cfg.VRD_NUM_RELA
N_round = cfg.VRD_TRAIN_ROUND
lr_init = cfg.VRD_LR_INIT
N_show = 100
N_save = 5000
N_val = 1000

index_sp = False
index_cls = False

roidb_path = cfg.DIR + 'vtranse/input/vrd_roidb.npz'
res_path = cfg.DIR + 'vtranse/pretrained_para/vgg_pretrained.ckpt'

roidb_read = read_roidb(roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']
N_train = len(train_roidb)
N_test = len(test_roidb)

i_temp = 4
roidb_use = train_roidb[i_temp]

im = cv2.imread(roidb_use['image'])
im = im[:,:,(2,1,0)]
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(im, aspect='equal')
sbox = roidb_use['sub_box_gt']
obox = roidb_use['obj_box_gt']
sb = roidb_use['sub_gt']
ob = roidb_use['obj_gt']
rela = roidb_use['rela_gt']
N_box = len(sbox)
for i in range(N_box):
	bbox = np.float32(sbox[i]) 
	ax.add_patch(
		plt.Rectangle((bbox[0], bbox[1]),
					  bbox[2] - bbox[0],
					  bbox[3] - bbox[1], fill=False,
					  edgecolor='red', linewidth=3)
		)
	ax.text(bbox[0], bbox[1] - 2,
				'{0}:{1} '.format(i,sb[i]),
				bbox=dict(facecolor='blue', alpha=0.5),
				fontsize=14, color='white')

	bbox = np.float32(obox[i]) 
	ax.add_patch(
		plt.Rectangle((bbox[0], bbox[1]),
					  bbox[2] - bbox[0],
					  bbox[3] - bbox[1], fill=False,
					  edgecolor='blue', linewidth=3)
		)
	ax.text(bbox[0], bbox[1] - 2,
				'{0}: {1} '.format(i,ob[i]),
				bbox=dict(facecolor='blue', alpha=0.5),
				fontsize=14, color='white')
	print('i:{0}, {1}-{2}-{3}'.format(i,sb[i],rela[i],ob[i]))

plt.axis('off')
plt.tight_layout()
plt.draw()
plt.show()