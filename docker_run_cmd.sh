#!/bin/bash
#mount the dataset and increase default shared memory size
sudo docker run --rm --shm-size=1g -v /nfs_3/data/cityscapes/:/nfs_3/data/cityscapes/ centaurausinfra/swin-transform-ss-cityscapes
