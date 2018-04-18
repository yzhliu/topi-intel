import nnvm.testing
import tvm
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import time

from tvm.contrib import graph_runtime
from schedule_pack.avx512_conv_fwd import *

out_shape = (1, )

def output_shape(in_size, kernel_num, kernel_size, pad, stride):
    out_size = (in_size + 2 * pad - kernel_size) // stride + 1
    return (1, kernel_num, out_size, out_size)


def output_shape_nChwc(shape):
    n, c, h, w = shape
    bn = 16 if c % 16 == 0 else c
    return (n, c // bn, h, w, bn)

def get_network():
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=7, strides=(2, 2), padding=(3, 3)))
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # # The Flatten layer collapses all axis, except the first one, into one axis.
        # net.add(gluon.nn.Flatten())
        # net.add(gluon.nn.Dense(num_fc, activation="relu"))
        # net.add(gluon.nn.Dense(num_outputs))
        net.add(gluon.nn.Activation(activation='sigmoid'))
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())
    net.hybridize()
    return net


def convert_layout_back(data):
    n, C, h, w, c = data.shape
    ic = C * c
    out = np.empty((n, ic, h, w))
    for a in range(n):
        for i in range(ic):
            for j in range(h):
                for k in range(w):
                    out[a, i, j, k] = data[a, i//c, j, k, i%c]
    return out


num_pass = 20
def end2end_benchmark(target, batch_size):
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    # out_shape = (batch_size, 64)
    out_shape = output_shape(in_size=224, kernel_num=64, kernel_size=7, pad=3, stride=2)

    block = get_network()
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")

    times = []
    for i in range(1):
        s = time.time()
        mxnet_out = block(mx.nd.array(data_array))
        mxnet_out.asnumpy()
        mkl_time = time.time() - s
        times.append(mkl_time)
    print("MKL inference time for batch size of %d: %f" % (batch_size, np.mean(times) * 1000))

    net, params = nnvm.frontend.from_mxnet(block)
    ctx = tvm.cpu()
    opt_level = 2
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)
    with open('graph.json', 'w') as fn:
        fn.writelines(graph.json())
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    input_data = tvm.nd.array(data_array, ctx=ctx)
    module.set_input('data', input_data)
    times = []
    for i in range(num_pass):
        s = time.time()
        module.run()
        tvm_time = time.time() - s
        times.append(tvm_time)
    print("TVM inference time for batch size of %d: %f" % (batch_size, np.mean(times) * 1000))
    tvm_out = module.get_output(0, out=tvm.nd.empty(output_shape_nChwc(out_shape)))

    np.testing.assert_array_almost_equal(convert_layout_back(tvm_out.asnumpy()), mxnet_out.asnumpy(), decimal=2)
    return tvm_time, mkl_time


if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    # KMP_AFFINITY=granularity=fine,compact,1,0 TVM_NUM_THREADS=16 OMP_NUM_THREADS=16 python test_topi_dev.py
    batch_size = 1
    # target = "llvm"
    # target = "llvm -mcpu=core-avx2"
    target = 'llvm -mcpu=skylake-avx512' # export TVM_NUM_THREADS=4 on c5xlarge
    tm, mm = end2end_benchmark(target, batch_size)
    print(tm, mm)