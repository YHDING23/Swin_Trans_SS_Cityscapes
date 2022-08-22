#!/bin/bash

#use cachebust argument to excude cache use for git repo
sudo docker build -t centaurausinfra/swin-transform-ss-cityscapes --build-arg CACHEBUST=$(date +%s) .
