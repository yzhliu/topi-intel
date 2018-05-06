import numpy as np
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model

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

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)#['softmax_label',])
    mod.bind(for_training=False, data_shapes=[('data', data_shape)])
             # label_shapes=data_iter.provide_label)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    # acc = mx.metric.create('acc')
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

        input = mx.io.DataBatch([normalized,], label=batch.label)
        mod.forward(input, is_train=False)
        pred = mx.nd.softmax(mod.get_outputs()[0])
        acc.update(batch.label, [pred])
        if i % 100 == 0:
            print(acc.get())

if __name__ == '__main__':
    end2end_benchmark('resnet152_v1', 1)