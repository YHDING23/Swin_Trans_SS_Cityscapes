# build cmd: docker build -t centaurusinfra/swin-transform-ss-cityscapes --build-arg CACHEBUST=$(date +%s) .
From pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install -y git vim wget
RUN apt-get install ffmpeg libsm6 libxext6  -y # for libGL.so in container

ARG CACHEBUST=1 # add this for not using cached git version during build
RUN git clone https://github.com/Fizzbb/Swin_Trans_SS_Cityscapes.git
WORKDIR ./Swin_Trans_SS_Cityscapes/

RUN pip install -r requirement.txt
RUN pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
RUN python setup.py develop
RUN wget https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r50_512x1024_40k_cityscapes/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth
RUN python model_verify.py

#CMD ["sleep", "infinity"]
CMD ["tools/dist_train.sh", "configs/upernet/upernet_r101_512x1024_40k_cityscapes.py", "2", "--options", "model.pretrained=upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth"]
