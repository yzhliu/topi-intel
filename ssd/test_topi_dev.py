import mxnet as mx
import numpy as np
import time
from collections import namedtuple
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime

from symbol.symbol_factory import get_symbol
from schedule_pack.avx512_conv_fwd import *

Batch = namedtuple('Batch', ['data'])
num_pass = 1000
def end2end_benchmark(body_network, target, batch_size):
    image_shape = (3, 512, 512)
    data_shape = (batch_size,) + image_shape
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")

    _, arg_params, aux_params = mx.model.load_checkpoint('model/ssd_resnet50_512', 0)
    sym = get_symbol(body_network, 512, num_classes=20)

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(data_shapes=[('data', data_shape)])
    mod.set_params(arg_params, aux_params)

    mx_data = mx.nd.array(data_array)
    times = []
    for i in range(20):
        s = time.time()
        mod.forward(Batch(data=[mx_data]), is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()
        mkl_time = time.time() - s
        times.append(mkl_time)
    print("MKL SSD inference time for batch size of %d: %f" % (batch_size, np.mean(times) * 1000))

    mxnet_out = mod.get_outputs()[0]
    out_shape = mxnet_out.shape
    print(out_shape)

    net, params = nnvm.frontend.from_mxnet(sym, mod.get_params()[0], mod.get_params()[1])

    ctx = tvm.cpu()
    opt_level = 3
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
    print("TVM %s inference time for batch size of %d: %f" % (body_network, batch_size, np.mean(times) * 1000))
    tvm_out = module.get_output(0, out=tvm.nd.empty(out_shape))

    # decimal=3 does not work for resnet-101
    np.testing.assert_array_almost_equal(tvm_out.asnumpy(), mxnet_out.asnumpy(), decimal=2)


if __name__ == "__main__":
    # KMP_AFFINITY=granularity=fine,compact,1,0 TVM_NUM_THREADS=16 OMP_NUM_THREADS=16 python test_topi_dev.py
    batch_size = 1
    # target = "llvm"
    # target = "llvm -mcpu=core-avx2"
    target = 'llvm -mcpu=skylake-avx512'
    end2end_benchmark('resnet50', target, batch_size)
