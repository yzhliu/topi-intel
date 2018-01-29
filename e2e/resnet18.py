import nnvm.testing
import tvm
import mxnet as mx
import numpy as np
import time

from tvm.contrib import graph_runtime
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model

def end2end_benchmark(model, target, batch_size):
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)

    block = get_model(model, pretrained=True)
    net, params = nnvm.frontend.from_mxnet(block)

    ctx = tvm.cpu()
    opt_level = 0
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    input_data = tvm.nd.array(data_array, ctx=ctx)
    module.set_input('data', input_data)

    s = time.time()
    module.run()
    tvm_time = time.time() - s
    print("TVM %s inference time for batch size of %d: %f" % (model, batch_size, tvm_time))
    tvm_out = module.get_output(0, out=tvm.nd.empty(out_shape))


    s = time.time()
    mxnet_out = block(mx.nd.array(data_array))
    mxnet_out.asnumpy()
    mkl_time = time.time() - s
    print("MKL %s inference time for batch size of %d: %f" % (model, batch_size, mkl_time))

    np.testing.assert_array_almost_equal(tvm_out.asnumpy(), mxnet_out.asnumpy(), decimal=3)

    return tvm_time, mkl_time


if __name__ == "__main__":
    batch_size = 1
    target = "llvm -mcpu=core-avx2"
    # tm, mm = end2end_benchmark('mobilenet1.0', target, batch_size)
    tm, mm = end2end_benchmark('resnet18_v1', target, batch_size)
    print(tm, mm)
