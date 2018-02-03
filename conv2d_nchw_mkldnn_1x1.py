import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple

# device = 'llvm -mcpu=core-avx2'
device = 'llvm -mcpu=skylake-avx512'
# device = 'llvm -mattr=+avx2'
# device = 'llvm'

dtype = 'float32'

ic_bn, oc_bn = 16, 16

# W10
# g1mb1_ic256ih14iw14_oc512oh7ow7_kh1kw1_sh2sw2_ph0pw0_n"resnet18_10"
batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 256, 14, 512, 1, 2, 0
oh_factor, ow_factor = 1, 7

# W7
# ic128ih28iw28_oc256oh14ow14_kh1kw1_sh2sw2_ph0pw0
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 128, 28, 256, 1, 2, 0
# oh_factor, ow_factor = 2, 14

# W4
# g1mb1_ic64ih56iw56_oc128oh28ow28_kh1kw1_sh2sw2_ph0pw0_n"resnet18_4"
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 64, 56, 128, 1, 2, 0
# oh_factor, ow_factor = 1, 28

# W2
# g1mb1_ic64ih56iw56_oc64oh56ow56_kh1kw1_sh1sw1_ph0pw0_n"resnet18_2"
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 64, 56, 64, 1, 1, 0
# oh_factor, ow_factor = 1, 28


def schedule_pack_data(input):
    # input: c, h, w
    osize = in_size + 2 * padding
    shape = (batch_size, in_channel//ic_bn, osize, osize, ic_bn)
    # shape = (batch_size, in_channel, osize, osize)
    # data_pad = tvm.compute(shape, lambda n, C, h, c, w: tvm.select(
    #     tvm.all(h >= padding, h < osize-padding, w >= padding, w < osize-padding),
    #     input[n, C*ic_bn+c, h-padding, w-padding], 0.0
    # ))
    data_pad = tvm.compute(shape, lambda n, C, h, w, c: input[n, C*ic_bn+c, h, w])
    # data_pad = tvm.compute(shape, lambda n, C, h, w, c: tvm.select(h < 14, input[n, C*ic_bn+c, h, w], 0.0))
    s = tvm.create_schedule(data_pad.op)
    return s, data_pad

def schedule_pack_kernel(input):
    # input: co, ci, h, w
    # output: gOIhw16i16o
    shape = (num_filter//oc_bn, in_channel//ic_bn, ic_bn, oc_bn)
    kernel_pack = tvm.compute(shape,
                              lambda CO, CI, ci, co: input[CO*oc_bn+co, CI*ic_bn+ci, 0, 0])
    s = tvm.create_schedule(kernel_pack.op)
    return s, kernel_pack

def schedule_conv(data, kernel):
    osize = (in_size + 2 * padding - kernel_size) // stride + 1
    # oshape = (batch_size, num_filter//oc_bn, 8, osize, oc_bn)
    oshape = (batch_size, num_filter//oc_bn, osize, osize, oc_bn)
    ovshape = (batch_size, num_filter//oc_bn, osize, oc_bn, osize)

    ic = tvm.reduce_axis((0, in_channel), name='ic')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
        tvm.sum(data[n, ic//ic_bn, oh*stride, ow*stride, ic%ic_bn].astype(dtype) *
                kernel[oc_chunk, ic//ic_bn, ic%ic_bn, oc_block],
                axis=[ic]),
        name='conv'
    )

    s = tvm.create_schedule(conv.op)
    C = conv
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=oh_factor)
    # ow_chunk, ow_block = s[C].split(ow, factor=ur_w)

    s[C].vectorize(oc_block)

    s[C].parallel(oc_chunk)
    s[C].pragma(batch, "parallel_launch_point")
    s[C].pragma(oc_chunk, "parallel_stride_pattern")
    s[C].pragma(batch, "parallel_barrier_when_finish")

    s[CC].compute_at(s[C], oh_outer)
    print(s[CC].op.axis)
    print(s[CC].op.reduce_axis)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)
    # s[CC].unroll(ic_block)

    return s, conv

def schedule_unpack_conv(input):
    osize = (in_size + 2 * padding - kernel_size) // stride + 1
    oshape = (batch_size, num_filter, osize, osize)
    unpack = tvm.compute(oshape, lambda n, oc, oh, ow: input[n, oc//oc_bn, oh, ow, oc%oc_bn])
    s = tvm.create_schedule(unpack.op)
    return s, unpack

def verify():
    ctx = tvm.context(device, 0)
    A = tvm.placeholder((batch_size, in_channel, in_size, in_size), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel_size, kernel_size), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

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
    print('data -> data_pack: %g ms/op' % (cost_data*1000.0))

    s, W_pack = schedule_pack_kernel(W)
    print(tvm.lower(s, [W, W_pack], simple_mode=True))
    w_pack = tvm.nd.array(np.zeros(get_const_tuple(W_pack.shape), dtype=dtype), ctx)
    func = tvm.build(s, [W, W_pack], device)
    time_f = func.time_evaluator(func.entry_name, ctx, number=1)
    cost_kernel = time_f(tvm.nd.array(w_np), w_pack,).mean
    print('kernel -> kernel_pack: %g ms/op' % (cost_kernel*1000.0))

    A_pack = tvm.placeholder(get_const_tuple(A_pack.shape), name='A_pack')
    W_pack = tvm.placeholder(get_const_tuple(W_pack.shape), name='W_pack')
    s, Conv = schedule_conv(A_pack, W_pack)
    print(tvm.lower(s, [A_pack, W_pack, Conv], simple_mode=True))
    conv = tvm.nd.array(np.zeros(get_const_tuple(Conv.shape), dtype=dtype), ctx)
    func = tvm.build(s, [A_pack, W_pack, Conv], device)
    time_f = func.time_evaluator(func.entry_name, ctx, number=10000)
    cost_conv_all = time_f(a_pack, w_pack, conv)
    # print(cost_conv_all)
    cost_conv = cost_conv_all.mean
    print('conv: %g ms/op' % (cost_conv*1000.0))
    func.save('conv.s')

    Conv = tvm.placeholder(get_const_tuple(Conv.shape), name='Conv_out')
    s, ConvUnpack = schedule_unpack_conv(Conv)
    print(tvm.lower(s, [Conv, ConvUnpack], simple_mode=True))
    conv_unpack = tvm.nd.array(np.zeros(get_const_tuple(ConvUnpack.shape), dtype=dtype), ctx)
    func = tvm.build(s, [Conv, ConvUnpack], device)
    time_f = func.time_evaluator(func.entry_name, ctx, number=1)
    cost_unpack = time_f(conv, conv_unpack).mean
    print('conv unpack: %g ms/op' % (cost_unpack*1000.0))

    # with tvm.build_config(auto_unroll_max_step=1400,
    #                       unroll_explicit=(device != "cuda")):
    print('expected shape: ' + str(conv_np.shape))
    print('unpack shape: ' + str(conv_unpack.asnumpy().shape))
    np.testing.assert_allclose(conv_unpack.asnumpy(), conv_np, rtol=1e-5)

if __name__ == "__main__":
    verify()
