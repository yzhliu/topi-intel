import numpy as np
import tvm

import topi
from topi.nn.conv2d import conv2d, _get_schedule
from topi.nn.conv2d import _get_workload

from topi.util import get_const_tuple, get_const_int
from topi.nn.pad import pad

from collections import namedtuple

AVX512Conv1x1Fwd = namedtuple('AVX512Conv1x1Fwd', ['ic_bn', 'oc_bn', 'oh_factor', 'ow_factor'])

def infer_stride(data, kernel, out):
    _, _, IH, IW, _ = data.shape
    CO, _, _, co, KH, KW = kernel.shape
    CO *= co
    _, _, OH, OW, _ = out.shape
    hstride = (IH - KH) // (OH - 1)
    wstride = (IW - KW) // (OW - 1)
    return get_const_int(hstride), get_const_int(wstride)

def infer_pad(data, data_pad):
    if data_pad is None:
        return 0, 0
    _, _, IH, IW, _ = data.shape
    _, _, TH, TW, _ = data_pad.shape
    hpad = (TH - IH) // 2
    wpad = (TW - IW) // 2
    return get_const_int(hpad), get_const_int(wpad)

def get_workload(data, kernel, stride, padding, out_dtype):
    """ Get the workload structure. """
    CO, CI, ci, co, KH, KW = [x.value for x in kernel.shape]
    ori_kernel = tvm.placeholder((CO*co, CI*ci, KH, KW))
    n, _, h, w, _ = [x.value for x in data.shape]
    original_data = tvm.placeholder((n, CI * ci, h, w))
    return _get_workload(original_data, ori_kernel, stride, padding, out_dtype)

def _declaration_conv(data, kernel, stride, padding, out_dtype):
    assert data.shape[0].value == 1, "only support batch size=1 convolution on rasp"
    wkl = get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size, in_channel_chunk, in_height, in_width, in_channel_block = get_const_tuple(data.shape)
    num_filter, _, _, co, kernel_height, kernel_width = get_const_tuple(kernel.shape)
    num_filter *= co

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

    # input: c, h, w
    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    in_channel = in_channel_block * in_channel_chunk
    if in_channel_block != sch.ic_bn:
        print('WARNING!!! (1x1) in_channel_block=%d vs sch.ic_bn=%d' % (in_channel_block, sch.ic_bn))
        shape = (batch_size, in_channel // sch.ic_bn, pad_height, pad_width, sch.ic_bn)
        data_vec = tvm.compute(shape, lambda n, C, h, w, c:
            data_pad[n, (C * sch.ic_bn + c) // in_channel_block, h, w, (C * sch.ic_bn + c) % in_channel_block],
                               tag='conv2d_data_pack')
    else:
        data_vec = data_pad

    kernel_pack = kernel

    oshape = (batch_size, num_filter // sch.oc_bn, out_height, out_width, sch.oc_bn)
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
        tvm.sum(data_vec[n, ic // sch.ic_bn, oh * HSTR, ow * WSTR, ic % sch.ic_bn].astype(out_dtype) *
                kernel_pack[oc_chunk, ic // sch.ic_bn, ic % sch.ic_bn, oc_block, 0, 0],
                axis=[ic]), name='conv2d_nChwc', tag='conv2d_nChwc')

    return conv


def _schedule_conv(s, data, data_pad, data_vec, kernel, conv_out, output, last):
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)

    wkl = get_workload(data, kernel, stride, padding, output.dtype)
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    DOPAD = (HPAD != 0 and WPAD != 0)

    # A, W = data, kernel_pack
    A0, A1 = data_pad, data_vec
    # schedule data
    if isinstance(s[A1].op, tvm.tensor.ComputeOp): # and  "conv2d_data_pack" in s[A1].op.tag:
        if DOPAD and "conv2d_data_pack" in s[A1].op.tag:
            s[A0].compute_inline()
        batch, ic_chunk, ih, iw, ic_block = s[A1].op.axis
        parallel_axis = s[A1].fuse(ic_chunk, ih)
        s[A1].parallel(parallel_axis)

    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[C].split(ow, factor=sch.ow_factor)
    s[C].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], oh_outer)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=sch.ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].vectorize(oc_block)

    if C == O:
        parallel_axis = s[CC].fuse(oc_chunk, oh_outer)
        s[CC].parallel(parallel_axis)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if O0 != O:
        s[O0].compute_inline()

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis

        # oc_chunk, oc_block = s[O].split(oc, factor=sch.oc_bn)
        oh_outer, oh_inner = s[O].split(oh, factor=sch.oh_factor)
        ow_outer, ow_inner = s[O].split(ow, factor=sch.ow_factor)
        s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

        parallel_axis = s[O].fuse(oc_chunk, oh_outer)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)

        s[O].parallel(parallel_axis)

    return s
