From pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install -y git vim

RUN git clone https://github.com/Fizzbb/Swin_Trans_SS_Cityscapes.git
WORKDIR ./Swin_Trans_SS_Cityscapes/
RUN ls

RUN pip install -r requirement.txt
RUN pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
RUN python setup.py develop
RUN wget https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r50_512x1024_40k_cityscapes/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth
RUN python model_verify.py

#CMD ["sleep", "infinity"]
CMD ["tools/dist_train.sh", "configs/upernet/upernet_r101_512x1024_40k_cityscapes.py", "8", "--options", "model.pretrained=upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth"]
