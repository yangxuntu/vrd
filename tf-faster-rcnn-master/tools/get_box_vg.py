from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net_vg
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import argparse
import pprint
import numpy as np
import sys

import tensorflow as tf
from nets.resnet_v1 import resnetv1

output_dir = '/home/yangxu/tf-faster-rcnn/output/res101/vg/vg_dete_pred/vg_pred_vg.npz'
model_path = '/home/yangxu/tf-faster-rcnn/output/res101/vg/default/res101_faster_rcnn_iter_165000.ckpt'



