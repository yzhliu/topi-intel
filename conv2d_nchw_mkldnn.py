import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from topi import tag
import scipy

# device = 'llvm -mcpu=core-avx2'
device = 'llvm -mcpu=skylake-avx512'
# device = 'llvm -mattr=+avx2'
# device = 'llvm'

dtype = 'float32'
batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 64, 56, 64, 3, 1, 1
bn = 16
ur_w = 28

def schedule_pack_data(input):
    # input: c, h, w
    osize = in_size + 2 * padding
    shape = (batch_size, in_channel//bn, osize, osize, bn)
    # shape = (in_channel, osize, osize)
    data_pad = tvm.compute(shape, lambda n, C, h, w, c: tvm.select(
        tvm.all(h >= padding, h < osize-padding, w >= padding, w < osize-padding),
        input[n, C*bn+c, h-padding, w-padding], 0.0
    ))
    s = tvm.create_schedule(data_pad.op)
    return s, data_pad

def schedule_pack_kernel(input):
    # input: co, ci, h, w
    # output: gOIhw16i16o
    shape = (num_filter//bn, in_channel//bn, kernel_size, kernel_size, bn, bn)
    kernel_pack = tvm.compute(shape,
                              lambda CO, CI, h, w, ci, co: input[CO*bn+co, CI*bn+ci, h, w])
    s = tvm.create_schedule(kernel_pack.op)
    return s, kernel_pack

def schedule_conv(data, kernel):
    osize = (in_size + 2 * padding - kernel_size) // stride + 1
    oshape = (batch_size, num_filter//bn, osize, osize, bn)

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_size), name='kh')
    kw = tvm.reduce_axis((0, kernel_size), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
        tvm.sum(data[n, ic//bn, oh*stride+kh, ow*stride+kw, ic%bn].astype(dtype) *
                kernel[oc_chunk, ic//bn, kh, kw, ic%bn, oc_block],
                axis=[ic, kh, kw]),
        name='conv'
    )

    s = tvm.create_schedule(conv.op)
    C = conv
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis

    ow_chunk, ow_block = s[C].split(ow, factor=ur_w)
    s[C].reorder(batch, oc_chunk, oh, ow_chunk, ow_block, oc_block)
    s[C].vectorize(oc_block)

    s[C].parallel(oc_chunk)
    s[C].pragma(batch, "parallel_launch_point")
    s[C].pragma(oc_chunk, "parallel_stride_pattern")
    s[C].pragma(batch, "parallel_barrier_when_finish")

    s[CC].compute_at(s[C], ow_chunk)
    print(s[CC].op.axis)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=ur_w)
    ic_chunk, ic_block = s[CC].split(ic, factor=bn)
    s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_block)
    # s[CC].unroll(ic_block)
    # s[CC].unroll(kw)

    return s, conv

def schedule_unpack_conv(input):
    osize = (in_size + 2 * padding - kernel_size) // stride + 1
    oshape = (batch_size, num_filter, osize, osize)
    unpack = tvm.compute(oshape, lambda n, oc, oh, ow: input[n, oc//bn, oh, ow, oc%bn])
    s = tvm.create_schedule(unpack.op)
    return s, unpack

def verify():
    ctx = tvm.context(device, 0)
    A = tvm.placeholder((batch_size, in_channel, in_size, in_size), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel_size, kernel_size), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    print(w_shape)

    @memoize("topi.tests.verify_conv_chw_mkldnn")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        conv_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, conv_np

    a_np, w_np, conv_np = get_ref_data()

    s, A_pack = schedule_pack_data(A)
    print(tvm.lower(s, [A, A_pack], simple_mode=True))
    a_pack = tvm.nd.array(np.zeros(get_const_tuple(A_pack.shape), dtype=dtype), ctx)
    func = tvm.build(s, [A, A_pack], device)
    time_f = func.time_evaluator(func.entry_name, ctx, number=1)
    cost_data = time_f(tvm.nd.array(a_np), a_pack).mean
    print('data -> data_pack: %g secs/op' % cost_data)

    s, W_pack = schedule_pack_kernel(W)
    print(tvm.lower(s, [W, W_pack], simple_mode=True))
    w_pack = tvm.nd.array(np.zeros(get_const_tuple(W_pack.shape), dtype=dtype), ctx)
    func = tvm.build(s, [W, W_pack], device)
    time_f = func.time_evaluator(func.entry_name, ctx, number=1)
    cost_kernel = time_f(tvm.nd.array(w_np), w_pack,).mean
    print('kernel -> kernel_pack: %g secs/op' % cost_kernel)

    A_pack = tvm.placeholder(get_const_tuple(A_pack.shape), name='A_pack')
    W_pack = tvm.placeholder(get_const_tuple(W_pack.shape), name='W_pack')
    s, Conv = schedule_conv(A_pack, W_pack)
    print(tvm.lower(s, [A_pack, W_pack, Conv], simple_mode=True))
    conv = tvm.nd.array(np.zeros(get_const_tuple(Conv.shape), dtype=dtype), ctx)
    func = tvm.build(s, [A_pack, W_pack, Conv], device)
    time_f = func.time_evaluator(func.entry_name, ctx, number=2000)
    cost_conv = time_f(a_pack, w_pack, conv).mean
    print('conv: %g sec/op' % cost_conv)
    func.save('conv.s')

    Conv = tvm.placeholder(get_const_tuple(Conv.shape), name='Conv_out')
    s, ConvUnpack = schedule_unpack_conv(Conv)
    print(tvm.lower(s, [Conv, ConvUnpack], simple_mode=True))
    conv_unpack = tvm.nd.array(np.zeros(get_const_tuple(ConvUnpack.shape), dtype=dtype), ctx)
    func = tvm.build(s, [Conv, ConvUnpack], device)
    time_f = func.time_evaluator(func.entry_name, ctx, number=1)
    cost_unpack = time_f(conv, conv_unpack).mean
    print('conv unpack: %g sec/op' % cost_unpack)

    # with tvm.build_config(auto_unroll_max_step=1400,
    #                       unroll_explicit=(device != "cuda")):
    print('expected shape: ' + str(conv_np.shape))
    print('unpack shape: ' + str(conv_unpack.asnumpy().shape))
    np.testing.assert_allclose(conv_unpack.asnumpy(), conv_np, rtol=1e-5)

if __name__ == "__main__":
    verify()
