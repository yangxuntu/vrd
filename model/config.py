from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.DIR = '/home/yangxu/project/rd/'

__C.DIR2 = '/home/jhmei/yangxu/'

__C.POOLING_SIZE = 7

#VG
############################
__C.VG_BATCH_NUM = 128
__C.VG_BATCH_NUM2 = 30
__C.VG_NUM_CLASS = 201
__C.VG_NUM_RELA = 100
__C.VG_LR_INIT = 0.00001
__C.VG_TRAIN_ROUND = 30
__C.VG_RBM_SPARSITY_LAMBDA = 0.001

__C.VG_BATCH_NUM_RELA = 128
__C.VG_AU_PAIR = 5
__C.VG_IOU_TRAIN = 0.5
__C.VG_IOU_TEST = 0.5
############################

#VRD
############################
__C.VRD_BATCH_NUM = 30
__C.VRD_NUM_CLASS = 101
__C.VRD_NUM_RELA = 70
__C.VRD_LR_INIT = 0.00001
__C.VRD_TRAIN_ROUND = 20

__C.VRD_BATCH_NUM_RELA = 50
__C.VRD_AU_PAIR = 5
__C.VRD_IOU_TRAIN = 0.5
__C.VRD_IOU_TEST = 0.5
###########################

__C.IM_SIZE = 600
__C.IM_MAX_SIZE = 1000

__C.TRAIN = edict()

__C.TRAIN.WEIGHT_DECAY = 0.0005


__C.RESNET = edict()

__C.RESNET.FIXED_BLOCKS = 1

__C.VTR = edict()

#the dimension of embedding space in vtranse
__C.VTR.VG_R = 500