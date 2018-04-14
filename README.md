# vtranse
visual translation embedding network for visual relation detection, CVPR 2017, tensorflow

1.What's inside?

Detector network: Faster RCNN

The detail of the implementation can be found from https://github.com/endernewton/tf-faster-rcnn, you can use the provided code in tf-faster-rcnn-master folder to train the detector network or download the pretrained model from https://share.weiyun.com/5skGi9N (vrd_vgg_pretrained.ckpt for vrd dataset and vg_vgg_pretrained.ckptfor vg dataset).

Vtranse network (includes predicate detection, phrase detection and relationship detection)
The detail of network can be found in CVPR2017 paper 'visual translation embedding network for visual relation detection, CVPR 2017, tensorflow'

2.Download links

The data and pre-trained model can be downloaded from https://share.weiyun.com/5skGi9N. The files include:

1). The processed data of visual relationship dataset (vrd_roidb.npz), the ra data can be downloaded from https://cs.stanford.edu/people/ranjaykrishna/vrd/.

2). The processed data of visual genome dataset (vg_roidb.npz), the raw data can be downloaded from https://visualgenome.org/.

3). The pretrained model of faster rcnn on vrd dataset(vrd_vgg_pretrained.ckpt) and vg dataset (vg_vgg_pretrained.ckpt)

3.Setup
1). Download the pretrained detection model from https://share.weiyun.com/5skGi9N and put them into the folder 'vtranse/pretrained_para'. And put vrd_roidb and vg_roidb in the folder 'vtranse/input'
2). Change the name 'DIR' in 'vtranse/model/cfg' file to suitable path.
3). Run 'python train_file/train_vrd_vgg.py' (or in ipython:  'run train_file/train_vrd_vgg.py') to train vtranse network by vrd dataset.
4). Run 'python test_file/test_vrd_vgg.py' (or in ipython:  'run test_file/test_vrd_vgg.py') and 'python test_file/eva_vrd_vgg_pred.py' to test and evaluate the result.

4.Citation

If you're using this code in a scientific publication please cite:

@inproceedings{Zhang_2017_CVPR,
  author    = {Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua},
  title     = {Visual Translation Embedding Network for Visual Relation Detection},
  booktitle = {CVPR},
  year      = {2017},
}

5.Reference

Vtrase Caffe type: https://github.com/zawlin/cvpr17_vtranse
Faster rcnn source code: https://github.com/endernewton/tf-faster-rcnn
