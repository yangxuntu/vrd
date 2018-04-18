# vtranse Tensorflow
visual translation embedding network for visual relation detection, CVPR 2017, tensorflow

1.What's inside?

Detector network: Faster RCNN

The details of the implementation can be found from https://github.com/endernewton/tf-faster-rcnn, you can use the provided code in tf-faster-rcnn-master folder to train the detector network or download the pretrained model from https://share.weiyun.com/5skGi9N (vrd_vgg_pretrained.ckpt for vrd dataset and vg_vgg_pretrained.ckptfor vg dataset).

Vtranse network (includes predicate detection, phrase detection and relationship detection)
The detail of network can be found in CVPR2017 paper 'visual translation embedding network for visual relation detection, CVPR 2017, tensorflow'

2.Download links

The data and pre-trained model can be downloaded from https://share.weiyun.com/5skGi9N. The files include:

1). The processed data of visual relationship dataset (vrd_roidb.npz), the raw data can be downloaded from https://cs.stanford.edu/people/ranjaykrishna/vrd/.

2). The processed data of visual genome dataset (vg_roidb.npz), the raw data can be downloaded from https://visualgenome.org/.

3). The pretrained model of faster rcnn on vrd dataset(vrd_vgg_pretrained.ckpt) and vg dataset (vg_vgg_pretrained.ckpt)

3.Setup

1). Download the pretrained detection model from https://share.weiyun.com/5skGi9N and put them into the folder 'vtranse/pretrained_para'. And put vrd_roidb and vg_roidb in the folder 'vtranse/input'

2). Change the name 'DIR' in 'vtranse/model/cfg' file to suitable path.

3). Run 'python train_file/train_vrd_vgg.py' (or in ipython:  'run train_file/train_vrd_vgg.py') to train vtranse network by vrd dataset.
```bash
python train_file/train_vrd_vgg.py
```
4). Run 'python test_file/test_vrd_vgg.py' (or in ipython:  'run test_file/test_vrd_vgg.py') and 'python test_file/eva_vrd_vgg_pred.py' to test and evaluate the result.
```bash
python test_file/test_vrd_vgg.py
python test_file/eva_vrd_vgg_pred.py
```


4. Training detection network by yourselves. If you do not want to train this network by yourself, you can download my pretrained file of faster-rcnn vrd dataset(vrd_vgg_pretrained.ckpt) and vg dataset (vg_vgg_pretrained.ckpt).

If you want to pretrain a detector network by yourself, you can:

1).Download raw data of VRD and VG from their offical web.

2).Download faster rcnn code from https://github.com/endernewton/tf-faster-rcnn and substitute the files in folder 'tf-faster-rcnn-master/lib' and 'tf-faster-rcnn-master/model' to my code.

3).Use the 'detector\vrd\vrd_process_dete.py' to generate the data 'vrd_roidb.npz' and put this source data into 'tf-faster-rcnn' folder.

4).Put the provided code 'vtranse/detector/vrd/train_vrd_dete_vgg.py' into 'tf-faster-rcnn-master/tools' and run this code in the root path of tf-faster-rcnn as:
```bash
python tools/train_vrd_dete_vgg.py
```

5.Train relationship detection by yousrself

1).Put the pretrained model into the folder 'vtranse/pretrained_para'

2).Use 'vtrnse\process\vrd_pred_process' to generate vrd_roidb.npz and put this file into 'vtranse\input'. If you do not want to generate this file, you can download them from the provided link. Notice that the provided 'vrd_roidb.npz' is used to train the relationship detection network and not used to train the detection network.

3).After putting the vrd_roidb.npz into vtranse\input, you should run 'python train_file/train_vrd_vgg.py'. And if you can not use python to run this file, the ipython is recommended. And in ipython, use:

```bash
run tools/train_vrd_dete_vgg.py
```

4).After training, use 'python test_file/test_vrd_vgg.py' to get the relationship result. You need to change the name of trained file in test_vrd_vgg.py to your trained model's name.

5).After getting the results, use 'python test_file/eva_vrd_vgg_pred.py' to evaluate the results.

6.Citation

If you're using this code in a scientific publication please cite:
```bash
@inproceedings{Zhang_2017_CVPR,
  author    = {Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua},
  title     = {Visual Translation Embedding Network for Visual Relation Detection},
  booktitle = {CVPR},
  year      = {2017},
}
```

7.Reference

1).Vtrase Caffe type: https://github.com/zawlin/cvpr17_vtranse

2).Faster rcnn source code: https://github.com/endernewton/tf-faster-rcnn

3).VRD dataset: https://cs.stanford.edu/people/ranjaykrishna/vrd/, https://drive.google.com/file/d/1BzP8DN2MAz76IvQTlpNOYla_bNC9gQuN/view

4).VG dataset: https://visualgenome.org/ . The filtered type of VG dataset https://drive.google.com/file/d/1C6MDiqWQupMrPOgk4T12zWiAJAZmY1aa/view
