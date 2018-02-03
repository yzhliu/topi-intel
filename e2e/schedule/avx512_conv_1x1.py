import numpy as np
import tvm

import topi
from topi.nn.conv2d import conv2d, _get_schedule
from topi.nn.conv2d import _get_workload

from topi.util import get_const_tuple
from topi.nn.util import infer_pad, infer_stride

from collections import namedtuple

AVX512Conv1x1Fwd = namedtuple('AVX512Conv1x1Fwd', ['ic_bn', 'oc_bn', 'oh_factor', 'ow_factor'])

def _declaration_conv(data, kernel, stride, padding, layout, out_dtype):
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

    # input: c, h, w
    shape = (batch_size, in_channel // sch.ic_bn, pad_height, pad_width, sch.ic_bn)
    data_vec = tvm.compute(shape, lambda n, C, h, w, c: data[n, C * sch.ic_bn + c, h, w])

    shape = (num_filter // sch.oc_bn, in_channel // sch.ic_bn, sch.ic_bn, sch.oc_bn, 1, 1)
    kernel_pack = tvm.compute(shape,
                              lambda CO, CI, ci, co, h, w: kernel[CO * sch.oc_bn + co, CI * sch.ic_bn + ci, h, w])

    oshape = (batch_size, num_filter // sch.oc_bn, out_height, out_width, sch.oc_bn)
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
        tvm.sum(data_vec[n, ic // sch.ic_bn, oh * HSTR, ow * WSTR, ic % sch.ic_bn].astype(out_dtype) *
                kernel_pack[oc_chunk, ic // sch.ic_bn, ic % sch.ic_bn, oc_block, 0, 0],
                axis=[ic]), name='conv')

    oshape = (batch_size, num_filter, out_height, out_width)
    unpack = tvm.compute(oshape, lambda n, oc, oh, ow: conv[n, oc // sch.oc_bn, oh, ow, oc % sch.oc_bn])
    return unpack


def _schedule_conv(s, data, data_pad, data_vec, kernel, kernel_pack, conv_out, output):
    # print('Run in avx512_conv_1x1 sch')
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
    if A0 is not None:
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

    C, O = conv_out, output
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=sch.oh_factor)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], oh_outer)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=sch.ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    batch, oc, oh, ow = s[O].op.axis

    oc_chunk, oc_block = s[O].split(oc, factor=sch.oc_bn)
    oh_outer, oh_inner = s[O].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[O].split(ow, factor=sch.ow_factor)
    s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

    # parallel_axis = s[O].fuse(oc_chunk, oh)
    parallel_axis = oc_chunk
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
        return AVX512Conv1x1Fwd(16, 16, 1, 28)

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
