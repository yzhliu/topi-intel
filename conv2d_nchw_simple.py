import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from topi import tag

def schedule_conv2d(outs):
    print('Run in x86 sch ...')
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'conv2d_nchw' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            C = conv
            print(C.op.axis)
            print(C.op.reduce_axis)
            print(data_pad.op.axis)

            n, c, h, w = C.op.axis
            rc, ry, rx = C.op.reduce_axis

            s[C].reorder(n, c, rc, h, w, ry, rx)
            r = s[C].fuse(ry, rx)
            s[C].unroll(r)

            xo, xi = s[C].split(w, factor=8)
            s[C].parallel(c)
            s[C].vectorize(xi)
            s[C].pragma(n, "parallel_launch_point")

    traverse(outs[0].op)
    return s

def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    def check_device():
        # print("Running on target: %s" % device)
        # device = 'llvm -mcpu=core-avx2'
        device = 'llvm -mcpu=skylake-avx512'
        # device = 'llvm -mattr=+avx2'
        # device = 'llvm'

        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
        W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
        B = topi.nn.conv2d(A, W, stride, padding)
        s = schedule_conv2d([B])
        print(tvm.lower(s, [A, W, B], simple_mode=True))

        a_shape = get_const_tuple(A.shape)
        w_shape = get_const_tuple(W.shape)
        dtype = A.dtype

        @memoize("topi.tests.test_topi_conv2d.verify_con2d_nchw")
        def get_ref_data():
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            w_np = np.random.uniform(size=w_shape).astype(dtype)
            b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
            c_np = np.maximum(b_np, 0)
            return a_np, w_np, b_np, c_np

        a_np, w_np, b_np, c_np = get_ref_data()
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=(device != "cuda")):
            func = tvm.build(s, [A, W, B], device, name="conv2d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding))

            time_f1 = func.time_evaluator(func.entry_name, ctx, number=400)
            cost = time_f1(a, w, b).mean
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            print('%g secs/op' % cost)

    check_device()


def test_conv2d_nchw():
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1)
    # ResNet18 worklaods
    """
    verify_conv2d_nchw(1, 3, 224, 64, 7, 3, 2)
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1)
    verify_conv2d_nchw(1, 64, 56, 64, 1, 1, 0)
    verify_conv2d_nchw(1, 64, 56, 128, 3, 2, 1)
    verify_conv2d_nchw(1, 64, 56, 128, 1, 2, 0)
    verify_conv2d_nchw(1, 128, 28, 128, 3, 1, 1)
    verify_conv2d_nchw(1, 128, 28, 256, 3, 2, 1)
    verify_conv2d_nchw(1, 128, 28, 256, 1, 2, 0)
    verify_conv2d_nchw(1, 256, 14, 256, 3, 1, 1)
    verify_conv2d_nchw(1, 256, 14, 512, 3, 2, 1)
    verify_conv2d_nchw(1, 256, 14, 512, 1, 2, 0)
    verify_conv2d_nchw(1, 512, 7, 512, 3, 1, 1)
    # Vgg16 workloads
    verify_conv2d_nchw(1, 128, 122, 128, 3, 1, 1)
    # Super resolution workloads
    verify_conv2d_nchw(1, 1, 224, 64, 5, 1, 2)
    verify_conv2d_nchw(1, 64, 224, 64, 3, 1, 1)
    verify_conv2d_nchw(1, 64, 224, 32, 3, 1, 1)
    verify_conv2d_nchw(1, 32, 224, 9, 3, 1, 1)
    """

if __name__ == "__main__":
    test_conv2d_nchw()
