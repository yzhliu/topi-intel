#!/usr/bin/env bash
for NUM_THREADS in {18..1}
do
    echo "nthreads = $NUM_THREADS"
    KMP_AFFINITY=granularity=fine,compact,1,0 TVM_NUM_THREADS=$NUM_THREADS OMP_NUM_THREADS=$NUM_THREADS \
    python test_topi_dev.py
    sleep 5
done