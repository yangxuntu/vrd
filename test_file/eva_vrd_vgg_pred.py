import numpy as np 
from model.config import cfg 
from model.ass_fun import *

gt_roidb_path = cfg.DIR + 'vtranse/input/vrd_roidb.npz'
pred_roidb_path = cfg.DIR + 'vtranse/pred_res/vrd_pred_roidb.npz'

roidb_read = read_roidb(gt_roidb_path)
train_roidb = roidb_read['train_roidb']
test_roidb = roidb_read['test_roidb']

roidb_read = read_roidb(pred_roidb_path)
pred_roidb = roidb_read['pred_roidb']

R50, num_right50 = rela_recall(test_roidb, pred_roidb, 50)
R100, num_right100 = rela_recall(test_roidb, pred_roidb, 100)

print('R50: {0}, R100: {1}'.format(R50, R100))