import numpy as np
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
import nnvm
import tvm
from tvm.contrib import graph_runtime

target = 'llvm -mcpu=skylake-avx512'
def end2end_benchmark(model, batch_size):
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)
    data_array = np.random.uniform(0, 255, size=data_shape).astype("float32")
    mx_data = mx.nd.array(data_array)

    block = get_model(model, pretrained=True)
    block.hybridize()
    block(mx_data)

    block.export("symbol/" + model)
    sym, arg_params, aux_params = mx.model.load_checkpoint("symbol/" + model, 0)

    data_iter = mx.io.ImageRecordIter(
        path_imgrec="/home/ubuntu/imagenet1k/imagenet1k-val.rec", # The target record file.
        data_shape=image_shape, # Output data shape; 227x227 region will be cropped from the original image.
        batch_size=batch_size, # Number of items per batch.
        # resize=256 # Resize the shorter edge to 256 before cropping.
        # You can specify more augmentation options. Use help(mx.io.ImageRecordIter) to see all the options.
    )

    net, params = nnvm.frontend.from_mxnet(block)

    ctx = tvm.cpu()
    opt_level = 3
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(net, target, shape={"data": data_shape}, params=params)
    with open('graph.json', 'w') as fn:
        fn.writelines(graph.json())
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)

    acc = mx.metric.TopKAccuracy(top_k=5)

    mean = np.array([[0.485, 0.456, 0.406]])
    std = np.array([[0.229, 0.224, 0.225]])
    mean = np.expand_dims(np.expand_dims(mean, 2), 2)
    std = np.expand_dims(np.expand_dims(std, 2), 2)

    i = 0
    for batch in data_iter:
        i += 1
        # You can now use the data_iter to access batches of images.
        # batch = data_iter.next() # first batch.
        images = batch.data[0] # This will contain 4 (=batch_size) images each of 3x227x227.
        normalized = images / 255
        normalized = mx.image.color_normalize(normalized,
                                              mean=mx.nd.array(mean),
                                              std=mx.nd.array(std))

        module.set_input('data', tvm.nd.array(normalized.asnumpy(), ctx=ctx))
        module.run()
        pred = module.get_output(0, out=tvm.nd.empty(out_shape))
        acc.update(batch.label, [mx.nd.array(pred.asnumpy())])
        if i % 100 == 0:
            print(acc.get())

if __name__ == '__main__':
    end2end_benchmark('resnet152_v1', 1)