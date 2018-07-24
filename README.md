# vtranse/STA Tensorflow
visual translation embedding network for visual relation detection, CVPR 2017, tensorflow

Shuffle-Then-Assemble: Learning Object-Agnostic Visual Relationship Features, ECCV, tensorflow

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
# Training and Testing Vtranse
## 1. Download dataset (VRD dataset is used as example)
a). Download the dataset form https://share.weiyun.com/55KK78Y, and the file is named as 'sg_dataset.zip'. 

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

## 2. Training
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

## 3. Testing
a). After training vtranse, you will find files like 'vrd_vgg0001.ckpt' in the 'vtranse/pred_para/vrd_vgg' folder. And
 then you can test your trained model
 
b). Open the file 'vtranse/test_file/test_vrd_vgg_pred.py' and change the variable 'model_path' to the suitable pretrained model's name.

c). Create a folder to save the result of detected relationships using the following commend:
```bash
mkdir -p ~vtranse/pred_res
```

d). After changing the name of your model, using following commend to get the relationship detection results:
```bash
ipython
run test_file/test_vrd_vgg_pred.py
```

e). After testing, you can run the file 'vtranse/test_file/eva_vrd_vgg_pred.py' to evaluate your detected result:
```bash
ipython
run test_file/eva_vrd_vgg_pred.py
```

# VG dataset
1). Download VG dataset.
This dataset can be downloaded from their offical website: https://visualgenome.org/. After downloading these files, you should using the following commend to put these images into the folder 'dataset/VG/images/VG_100K'
```bash
mkdir -p ~dataset/VG/images/VG_100K
mv images/VG100K dataset/VG/images/VG_100K
mv images/VG100K dataset/VG/images/VG_100K
```

2). Download training/testing split
Since this dataset is so noisy, and I use one filtered type which is provided by https://drive.google.com/file/d/1C6MDiqWQupMrPOgk4T12zWiAJAZmY1aa/view?usp=drive_web, you can download the split form this link.
After downloading this file, you can use the following commend to pre-process the vg dataset

```bash
mkdir -p ~dataset/VG/imdb
mv vg1_2_meta.h5 dataset/VG/imdb
ipython
run process/vg_pred_process.py
```

3). Training, Testing and Evaluation
After pre-processing Vg dataset, you can using similar process like VRD dataset to train, test and evaluate your model by using following commends:
```bash
ipython
run train_file/train_vg_vgg.py
```

```bash
ipython
run test_file/test_vg_vgg_pred.py
```

```bash
ipython
run test_file/eva_vg_vgg_pred.py
```

# Citation:

```bash
@inproceedings{Zhang_2017_CVPR,
  author    = {Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua},
  title     = {Visual Translation Embedding Network for Visual Relation Detection},
  booktitle = {CVPR},
  year      = {2017},
}
```

# Results of VRD (R100)
|               |  predicate    | phrase        | relation    |
| ------------- |:-------------:| -------------:| -----------:|
| published result        | 44.76      | 22.42   |15.20|
| implemented result      | 46.48      |   24.32 |16.27|


# Results of VG (R100)
|               |  predicate    | phrase        | relation    |
| ------------- |:-------------:| -------------:| -----------:|
| published result        | 62.87      | 10.45   |6.04|
| implemented result      | 61.70      | 13.62 |11.62|

# References:
1. VRD project:
https://cs.stanford.edu/people/ranjaykrishna/vrd/

2. Visual Genome
https://visualgenome.org

3. Vtranse Caffe Type:
https://github.com/zawlin/cvpr17_vtranse

4. The faster rcnn code which I used to train the detection part in this file:
https://github.com/endernewton/tf-faster-rcnn

# Contact
1. If you have any problems of this programming, you can eamil to s170018@e.ntu.edu.sg.
