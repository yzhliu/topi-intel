import mxnet as mx
import numpy as np
import time
import nnvm.testing
import tvm

from collections import namedtuple

from tvm.contrib import graph_runtime
from mxnet.gluon.model_zoo.vision import get_model

# Batch = namedtuple('Batch', ['data'])
num_pass = 1000
def end2end_benchmark(model, target, batch_size):
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    mx_data = mx.nd.array(data_array)

    block = get_model(model, pretrained=True)
    block.hybridize()
    block(mx_data)
    net, params = nnvm.frontend.from_mxnet(block)
    ctx = tvm.cpu()
    opt_level = 2
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    block.export("symbol/" + model)
    sym, arg_params, aux_params = mx.model.load_checkpoint("symbol/" + model, 0)
    args = dict(arg_params, **{'data': mx.nd.empty(data_shape)})
    exec_ = sym.bind(ctx=mx.cpu(), args=args, aux_states=aux_params)
    mx_data.copyto(exec_.arg_dict['data'])

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', data_shape)],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    times = []
    for i in range(num_pass):
        s = time.time()
        exec_.forward()
        for output in exec_.outputs:
            output.wait_to_read()
        mkl_time = time.time() - s
        times.append(mkl_time)
    mxnet_out = output
    print("MKL %s inference time for batch size of %d: %f" % (model, batch_size, np.mean(times) * 1000))


    input_data = tvm.nd.array(data_array, ctx=ctx)
    module.set_input('data', input_data)
    times = []
    for i in range(20):
        s = time.time()
        module.run()
        tvm_time = time.time() - s
        times.append(tvm_time)
    print("TVM %s inference time for batch size of %d: %f" % (model, batch_size, np.mean(times) * 1000))
    tvm_out = module.get_output(0, out=tvm.nd.empty(out_shape))

    np.testing.assert_array_almost_equal(tvm_out.asnumpy(), mxnet_out.asnumpy(), decimal=2)


if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    import sys

    batch_size = 1
    # target = "llvm"
    # target = "llvm -mcpu=core-avx2"
    target = 'llvm -mcpu=skylake-avx512' # export TVM_NUM_THREADS=4 on c5xlarge
    if len(sys.argv) == 2:
        end2end_benchmark(sys.argv[1], target, batch_size)
    else:
        # end2end_benchmark('mobilenet1.0', target, batch_size)
        # end2end_benchmark('resnet18_v1', target, batch_size)
        # end2end_benchmark('resnet34_v1', target, batch_size)
        end2end_benchmark('resnet50_v1', target, batch_size)
        # end2end_benchmark('resnet101_v1', target, batch_size)
        # end2end_benchmark('resnet152_v1', target, batch_size)
