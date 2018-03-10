import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import time

from tvm.contrib import graph_runtime
from mxnet.gluon.model_zoo.vision import get_model
from schedule_pack.avx512_conv_fwd import *

num_pass = 2000
def end2end_benchmark(model, target, batch_size):
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)

    block = get_model(model, pretrained=True)
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")

    times = []
    for i in range(1):
        s = time.time()
        mxnet_out = block(mx.nd.array(data_array))
        mxnet_out.asnumpy()
        mkl_time = time.time() - s
        times.append(mkl_time)
    print("MKL %s inference time for batch size of %d: %f" % (model, batch_size, np.mean(times) * 1000))

    # data_array_5D = np.expand_dims(data_array, 4)
    # print("data_array_5D shape: " + str(data_array_5D.shape))

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
    print("TVM %s inference time for batch size of %d: %f" % (model, batch_size, np.mean(times) * 1000))
    tvm_out = module.get_output(0, out=tvm.nd.empty(out_shape))

    # decimal=3 does not work for resnet-101
    np.testing.assert_array_almost_equal(tvm_out.asnumpy(), mxnet_out.asnumpy(), decimal=2)

    return tvm_time, mkl_time


if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    # KMP_AFFINITY=granularity=fine,compact,1,0 TVM_NUM_THREADS=16 OMP_NUM_THREADS=16 python test_topi_dev.py

    batch_size = 1
    # target = "llvm"
    # target = "llvm -mcpu=core-avx2"
    target = 'llvm -mcpu=skylake-avx512' # export TVM_NUM_THREADS=4 on c5xlarge
    # tm, mm = end2end_benchmark('mobilenet1.0', target, batch_size)
    # tm, mm = end2end_benchmark('resnet18_v1', target, batch_size)
    # tm, mm = end2end_benchmark('resnet34_v1', target, batch_size)
    # tm, mm = end2end_benchmark('resnet50_v1', target, batch_size)
    tm, mm = end2end_benchmark('resnet101_v1', target, batch_size)
    # tm, mm = end2end_benchmark('resnet152_v1', target, batch_size)
    print(tm, mm)
