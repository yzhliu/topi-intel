#!/usr/bin/env bash
for network in resnet18_v1 resnet18_v2 resnet34_v1 resnet34_v2 resnet50_v1 resnet101_v1 resnet152_v1
do
    for NUM_THREADS in 1 2 4 8 16 18
    do
        echo "network=$network nthreads=$NUM_THREADS"
        KMP_AFFINITY=granularity=fine,compact,1,0 TVM_NUM_THREADS=$NUM_THREADS OMP_NUM_THREADS=$NUM_THREADS \
        python test_topi_dev.py $network
        sleep 5
    done
done