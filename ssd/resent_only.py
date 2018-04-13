import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import time
from collections import namedtuple

from symbol.resnet import get_symbol

from tvm.contrib import graph_runtime
from mxnet.gluon.model_zoo.vision import get_model
from schedule_pack.avx512_conv_fwd import *

Batch = namedtuple('Batch', ['data'])
num_pass = 200
def end2end_benchmark(model, target, batch_size):
    num_classes = 20
    image_shape = (3, 512, 512)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)

    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")

    sym = get_symbol(num_classes=num_classes, num_layers=50, image_shape='3,224,224')
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(data_shapes=[('data', data_shape)])
    mod.init_params()

    # print(len(mod.get_params()))

    times = []
    mx_in_data = mx.nd.array(data_array)
    for i in range(num_pass):
        s = time.time()
        mod.forward(Batch(data=[mx_in_data]), is_train=False)
        mxnet_out = mod.get_outputs()[0]
        # mxnet_out = block(mx.nd.array(data_array))
        # mxnet_out.asnumpy()
        mxnet_out.wait_to_read()
        mkl_time = time.time() - s
        times.append(mkl_time)
    print("MKL %s inference time for batch size of %d: %f" % (model, batch_size, np.mean(times) * 1000))

    net, params = nnvm.frontend.from_mxnet(sym, mod.get_params()[0], mod.get_params()[1])
    ctx = tvm.cpu()
    opt_level = 3
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target,
                                                 shape={"data": data_shape},
                                                 params=params)
                                                 # layout="NCHW")
    with open('graph.json', 'w') as fn:
        fn.writelines(graph.json())
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    input_data = tvm.nd.array(data_array, ctx=ctx)
    module.set_input('data', input_data)

    # warm up
    for i in range(100):
        module.run()

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
    # KMP_AFFINITY=granularity=fine,compact,1,0 TVM_NUM_THREADS=16 OMP_NUM_THREADS=16 python test_topi_dev.py
    batch_size = 1
    # target = "llvm"
    # target = "llvm -mcpu=core-avx2"
    target = 'llvm -mcpu=skylake-avx512'
    # tm, mm = end2end_benchmark('mobilenet1.0', target, batch_size)
    # tm, mm = end2end_benchmark('resnet18_v2', target, batch_size)
    # tm, mm = end2end_benchmark('resnet34_v2', target, batch_size)
    tm, mm = end2end_benchmark('resnet50_v2', target, batch_size)
    # tm, mm = end2end_benchmark('resnet101_v1', target, batch_size)
    # tm, mm = end2end_benchmark('resnet152_v1', target, batch_size)
    print(tm, mm)
