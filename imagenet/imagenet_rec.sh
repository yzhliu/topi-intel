#!/usr/bin/env bash
wget http://data.mxnet.io/models/imagenet/resnet/val.lst -O imagenet1k-val.lst
python im2rec.py --resize 224 --quality 90 --num-thread 18 imagenet1k-val val/