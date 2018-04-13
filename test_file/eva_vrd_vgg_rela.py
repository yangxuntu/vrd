import numpy as np 
from model.config import cfg 
from model.ass_fun import *

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