## An Implementation of Swin-Transformer-Semantic-Segmentation
This repo contains the supported code and configuration files to reproduce semantic segmentation results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). You can find the original code from [here](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation). 

### Step 1. Installation of `mmsegmentation`
Swin-Trans-Sem-Seg code is heavily based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0), and the original repo of `mmsegmentation` installation is [here](https://github.com/open-mmlab/mmsegmentation/blob/v0.11.0/docs/get_started.md#installation). Hope this repo can simplify your setting-up. 

a. Virtual environment:
```angular2html
sudo apt install python3.7 # swin-trans require python>=3.6
git clone https://github.com/YHDING23/Swin_Trans_SS_Cityscapes.git
cd Swin_Trans_SS_Cityscapes
mkdir venv
virtualenv -p python3.7 venv
source venv/bin/activate
```

b. Requirements
```angular2html
pip install -r requirement.txt
```

c. Install mmcv w.r.t. your cuda version. 
Check your cuda version. Open you python interpreter and:
```angular2html
import torch
torch.cuda.is_available() # Ture, otherwise GPU is not available
torch.version.cuda # print out your cuda version.
```
For instance, your cuda version is 10.2, then go back to the shell and 
```angular2html
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html
```
Please note the mcvv-full version is critical and may cause issues. If the error suggests a lower or higher mmcv version, please first uninstall the current mmcv then re-install the suggested version accordingly. 
```angular2html
pip uninstall mmcv-full
```

d. Install mmsegmentation
```angular2html
python setup.py develop
```

e. Verify

First download a checkpoint file, e.g. :
```angular2html
wget https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r50_512x1024_40k_cityscapes/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth
```
The model corresponds to the config file in `configs/upernet/upernet_r101_512x1024_40k_cityscapes.py`.
Then, Open you python interpreter and run the following codes.
```
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'configs/upernet/upernet_r101_512x1024_40k_cityscapes.py'
checkpoint_file = 'upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')
```
You can visualizae the segmentation results in `result.jpg`.  

### Step 2. Prepare Dataset and Training
We prefer the cityscapes dataset for training. There is a copy in our NFS server `/nfs_3/data/cityscapes`. Change the `data_root` in `configs/_base_/datasets/cityscapes.py` to:
```angular2html
data_root='/nfs_3/data/cityscapes'
```

Download a pretrained model which is similar to the `upernet` checkpoint file used for the above verification. A model zoo can be found [here](https://github.com/open-mmlab/mmsegmentation/blob/v0.11.0/docs/model_zoo.md). Make sure to download the corresponding config file as well. 
Start training:
```angular2html
tools/dist_train.sh configs/upernet/upernet_r101_512x1024_40k_cityscapes.py 8 --options model.pretrained=pretrain/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth
```

