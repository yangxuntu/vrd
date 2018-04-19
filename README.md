# vtranse Tensorflow
visual translation embedding network for visual relation detection, CVPR 2017, tensorflow

# Installation
1. Install ipython, if you do not have ipython, you can install this tool (strongly recommended: https://ipython.org/install.html)
```bash
pip install ipython
```
2. Install TensorFlow v1.3.0 or newer type.
```bash
pip install tensorflow-gpu==1.3.0
```
3.Download this repository or clone with Git
```bash
git clone https://github.com/yangxuntu/vtranse.git
```

3. Install easydict
```bash
pip install easydict
```

# 1. Download dataset (VRD dataset is used as example)
a). Download the dataset form https://share.weiyun.com/5skGi9N, and the file is named as 'sg_dataset.zip'. 

b). Use the following commend to unzip the downloaded data:
```bash
unzip sg_dataset.zip -d sg_dataset
```

c).In the path where you put vtranse folder, use the following commend to make a new folder 'dataset/VRD':
```bash
mkdir -p ~/dataset/VRD/json_dataset
mkdir -p ~/dataset/VRD/sg_dataset
```

d). Move the files in sg_dataset into the created dataset, by using the following commends:
```bash
mv sg_dataset/annotations_test.json dataset/VRD/json_dataset
mv sg_dataset/annotations_train.json dataset/VRD/json_dataset
mv sg_dataset/sg_test_images dataset/VRD/sg_dataset
mv sg_dataset/sg_train_images dataset/VRD/sg_dataset
```

e). Change the root path in file 'vtranse/model/config.py': open this file and find the term '__C.DIR' which is named as '/home/yangxu/rd' to suitable path where you put this vtrase folder. 

f). Pre-process the VRD dataset to the vrd_roidb.npz which can be used to train the network. Open ipython using following commend:
```bash
ipython
```
And then use following commend to pre-process data in vrd folder:
```bash
run process/vrd_pred_process.py
```

After runing this file, you will find that there is one 'vrd_roidb.npz' file in the foloder 'vtranse/input'

# 2. Training
a). Download pre-trained model of faster-rcnn on VRD dataset from https://share.weiyun.com/5skGi9N, and the file names are 'vrd_vgg_pretrained.ckpt.data-00000-of-00001', 'vrd_vgg_pretrained.ckpt.index', 'vrd_vgg_pretrained.ckpt.meta' and 'vrd_vgg_pretrained.ckpt.pkl'. After downloading them, using the following commend to move them into the 'vtranse/pre_trained' file:
```bash
mv vrd_vgg_pretrained.ckpt.data-00000-of-00001 vtranse/pretrained_para
mv vrd_vgg_pretrained.ckpt.index vtranse/pretrained_para
mv vrd_vgg_pretrained.ckpt.meta vtranse/pretrained_para
mv vrd_vgg_pretrained.ckpt.pkl vtranse/pretrained_para
```

b). Create a folder which is used to save the trained results
```bash
mkdir -p ~vtranse/pred_para/vrd_vgg
```

c). After downloading and moving files to suitable folder, using 'vtranse/train_file/train_vrd_vgg.py' to train vtranse network on VRD dataset.
```bash
ipython
run train_file/train_vrd_vgg.py
```
d). When training, you can see the results like that:
```bash
t: 100.0, rd_loss: 4.83309404731, acc: 0.0980000074953
t: 200.0, rd_loss: 3.81237616211, acc: 0.263000019006
t: 300.0, rd_loss: 3.51845422685, acc: 0.290333356783
t: 400.0, rd_loss: 3.31810754955, acc: 0.292666691653
t: 500.0, rd_loss: 3.48527273357, acc: 0.277666689083
t: 600.0, rd_loss: 3.06100189149, acc: 0.340666691475
t: 700.0, rd_loss: 3.02625158072, acc: 0.334666692317
t: 800.0, rd_loss: 3.06034492403, acc: 0.330333357863
t: 900.0, rd_loss: 3.16739703059, acc: 0.322666690871
...
```

# 3. Testing







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
