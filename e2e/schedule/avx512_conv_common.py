from __future__ import absolute_import as _abs
import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from collections import namedtuple

from topi.nn.conv2d import conv2d, _get_schedule
from topi.nn.conv2d import _get_workload
from topi.nn.util import infer_pad, infer_stride

AVX512ConvCommonFwd = namedtuple('AVX512ConvCommonFwd', ['ic_bn', 'oc_bn', 'ur_w', 'unroll_kw'])

def _declaration_conv(data, kernel, stride, padding, layout, out_dtype):
    # print('Run in avx512_conv_common decl')
    assert layout == 'NCHW', "only support NCHW convolution on rasp"
    assert data.shape[0].value == 1, "only support batch size=1 convolution on rasp"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

    # pack data
    # input: c, h, w
    shape = (batch_size, in_channel, pad_height, pad_width)
    data_pad = tvm.compute(shape, lambda n, c, h, w: tvm.select(
        tvm.all(h >= HPAD, h < pad_height - HPAD, w >= WPAD, w < pad_width - WPAD),
        data[n, c, h - HPAD, w - WPAD], 0.0
    ), name='data_pad')

    shape = (batch_size, in_channel // sch.ic_bn, pad_height, sch.ic_bn, pad_width)
    data_vec = tvm.compute(shape,
                           lambda n, C, h, c, w: data_pad[n, C * sch.ic_bn + c, h, w],
                           name='data_vec')

    # pack kernel
    # input: co, ci, h, w
    # output: gOIhw16i16o
    shape = (num_filter // sch.oc_bn, in_channel // sch.ic_bn, kernel_height, kernel_width, sch.ic_bn, sch.oc_bn)
    kernel_pack = tvm.compute(shape,
                              lambda CO, CI, h, w, ci, co: kernel[CO * sch.oc_bn + co, CI * sch.ic_bn + ci, h, w],
                              name='kernel_pack')

    # convolution
    oshape = (batch_size, num_filter // sch.oc_bn, out_height, out_width, sch.oc_bn)
    ovshape = (batch_size, num_filter // sch.oc_bn, out_height, sch.oc_bn, out_width)
    unpack_shape = (batch_size, num_filter, out_height, out_width)

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
        tvm.sum(data_vec[n, ic // sch.ic_bn, oh * HSTR + kh, ic % sch.ic_bn, ow * WSTR + kw].astype(out_dtype) *
                kernel_pack[oc_chunk, ic // sch.ic_bn, kh, kw, ic % sch.ic_bn, oc_block],
                axis=[ic, kh, kw]), name='conv')

    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, c // sch.oc_bn, h, w, c % sch.oc_bn])
    return unpack


def _schedule_conv(s, data, data_pad, data_vec, kernel, kernel_pack, conv_out, output):
    # print('Run in avx512_conv_common sch')
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)
    wkl = _get_workload(data, kernel, stride, padding, output.dtype)
    sch = _get_schedule(wkl)

    A, W = data, kernel_pack
    A0, A1 = data_pad, data_vec
    # schedule data
    s[A0].compute_inline()
    batch, ic_chunk, ih, ic_block, iw = s[A1].op.axis
    parallel_axis = s[A1].fuse(ic_chunk, ih)
    s[A1].parallel(parallel_axis)
    s[A1].pragma(batch, "parallel_launch_point")
    s[A1].pragma(parallel_axis, "parallel_stride_pattern")
    s[A1].pragma(batch, "parallel_barrier_when_finish")

    # schedule kernel pack
    oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
    if sch.oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, oh)
    s[W].parallel(parallel_axis)
    s[W].pragma(parallel_axis, "parallel_launch_point")
    s[W].pragma(parallel_axis, "parallel_stride_pattern")
    s[W].pragma(parallel_axis, "parallel_barrier_when_finish")

    # schedule conv
    C, O = conv_out, output
    CC = s.cache_write(C, 'global')

    _, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=sch.ur_w)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=sch.ur_w)
    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    if sch.unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].fuse(oc_chunk, oh)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_block)

    batch, oc, oh, ow = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=sch.ur_w)
    oc_chunk, oc_block = s[O].split(oc, factor=sch.oc_bn)
    s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(oc_chunk, oh)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)

    s[O].parallel(parallel_axis)
    s[O].pragma(batch, "parallel_launch_point")
    s[O].pragma(parallel_axis, "parallel_stride_pattern")
    s[O].pragma(batch, "parallel_barrier_when_finish")

    return s

if __name__ == "__main__":
    device = 'llvm -mcpu=skylake-avx512'
    dtype = 'float32'

    # W2
    batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 64, 56, 64, 1, 1, 0

    @_get_schedule.register("cpu", override=True)
    def _get_schedule_conv(wkl):
        return AVX512ConvCommonFwd(16, 16, 28, False)

    ctx = tvm.context(device, 0)
    A = tvm.placeholder((batch_size, in_channel, in_size, in_size), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel_size, kernel_size), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        conv_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, conv_np

    a_np, w_np, conv_np = get_ref_data()

    with tvm.target.create(device):
        Conv = _declaration_conv(A, W, stride, padding, 'NCHW', dtype)
        s = tvm.create_schedule(Conv.op)

        op = Conv.op
        output = op.output(0)
        conv_out = op.input_tensors[0]

        kernel_pack = conv_out.op.input_tensors[1]
        kernel = kernel_pack.op.input_tensors[0]

        data_vec = conv_out.op.input_tensors[0]
        data = data_vec.op.input_tensors[0]
        data_pad = None
        if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.name:
            data_pad = data
            data = data_pad.op.input_tensors[0]

        s = _schedule_conv(s, data, data_pad, data_vec, kernel, kernel_pack, conv_out, output)
        print(tvm.lower(s, [A, W, Conv], simple_mode=True))
        conv_unpack = tvm.nd.array(np.zeros(get_const_tuple(Conv.shape), dtype=dtype), ctx)
        func = tvm.build(s, [A, W, Conv], device)
        time_f = func.time_evaluator(func.entry_name, ctx, number=2000)
        cost_unpack = time_f(tvm.nd.array(a_np), tvm.nd.array(w_np), conv_unpack).mean
        print('conv: %g ms/op' % (cost_unpack * 1000.0))

# W0
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 3, 224, 64, 7, 2, 3
# ic_bn, oc_bn, ur_w = 3, 16, 28
# verify(1, 3, 224, 64, 7, 2, 3)

# W1
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 64, 56, 64, 3, 1, 1
# ic_bn, oc_bn, ur_w = 16, 16, 28
# verify(1, 64, 56, 64, 3, 1, 1)

# W2
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 64, 56, 64, 1, 1, 0
# ic_bn, oc_bn, ur_w = 16, 16, 28
# verify(1, 64, 56, 64, 1, 1, 0)

# W3
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 64, 56, 128, 3, 2, 1
# ic_bn, oc_bn, ur_w = 16, 16, 28
# verify(1, 64, 56, 128, 3, 2, 1)

# W4
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 64, 56, 128, 1, 2, 0
# ic_bn, oc_bn, ur_w = 16, 16, 28
# verify(1, 64, 56, 128, 1, 2, 0)

# W5
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 128, 28, 128, 3, 1, 1
# ic_bn, oc_bn, ur_w = 16, 16, 28
# verify(1, 128, 28, 128, 3, 1, 1)

# W6
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 128, 28, 256, 3, 2, 1
# ic_bn, oc_bn, ur_w = 16, 16, 14
# verify(1, 128, 28, 256, 3, 2, 1)

# W7
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 128, 28, 256, 1, 2, 0
# ic_bn, oc_bn, ur_w = 16, 16, 14
# verify(1, 128, 28, 256, 1, 2, 0)

# W8
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 256, 14, 256, 3, 1, 1
# ic_bn, oc_bn, ur_w, unroll_kw = 16, 16, 14, True
# verify(1, 256, 14, 256, 3, 1, 1)

# W9
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 256, 14, 512, 3, 2, 1
# ic_bn, oc_bn, ur_w, unroll_kw = 16, 32, 7, True
# verify(1, 256, 14, 512, 3, 2, 1)

# W10
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 256, 14, 512, 1, 2, 0
# ic_bn, oc_bn, ur_w = 16, 32, 7
# verify(1, 256, 14, 512, 1, 2, 0)

# W11
# batch_size, in_channel, in_size, num_filter, kernel_size, stride, padding = 1, 512, 7, 512, 3, 1, 1
# ic_bn, oc_bn, ur_w, unroll_kw = 16, 16, 7, True
# verify(1, 512, 7, 512, 3, 1, 1)
