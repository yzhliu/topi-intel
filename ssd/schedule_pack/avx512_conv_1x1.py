import numpy as np
import tvm

import topi
from topi.nn.conv2d import conv2d, _get_schedule
from topi.nn.conv2d import _get_workload

from topi.util import get_const_tuple, get_const_int
from topi.nn.pad import pad

from collections import namedtuple

AVX512Conv1x1Fwd = namedtuple('AVX512Conv1x1Fwd',
                              ['ic_bn', 'oc_bn', 'oh_factor', 'ow_factor', 'layout_in', 'layout_out'])


def _declaration_conv(wkl, data, kernel):
    assert data.shape[0].value == 1, "only support batch size=1 convolution on rasp"
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    ndim_input = len(data.shape)

    if ndim_input == 5:
        batch_size, in_channel_chunk, in_height, in_width, in_channel_block = get_const_tuple(data.shape)
        in_channel = in_channel_block * in_channel_chunk
    else:
        assert ndim_input == 4
        in_channel_block = 0
        batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)

    num_filter, _, _, co, kernel_height, kernel_width = get_const_tuple(kernel.shape)
    num_filter *= co

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

    # input: c, h, w
    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        if ndim_input == 5:
            data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
        else:
            assert ndim_input == 4
            data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    if in_channel_block != sch.ic_bn:
        print('WARNING!!! (1x1) in_channel_block=%d vs sch.ic_bn=%d' % (in_channel_block, sch.ic_bn))
        shape = (batch_size, in_channel // sch.ic_bn, pad_height, pad_width, sch.ic_bn)
        if ndim_input == 5:
            data_vec = tvm.compute(shape,
                                   lambda n, C, h, w, c:
                                   data_pad[n, (C * sch.ic_bn + c) // in_channel_block, h, w, (C * sch.ic_bn + c) % in_channel_block],
                                   name='data_vec', tag="conv2d_data_pack")
        else:
            assert ndim_input == 4
            data_vec = tvm.compute(shape,
                                   lambda n, C, h, w, c:
                                   data_pad[n, (C * sch.ic_bn + c), h, w],
                                   name='data_vec', tag="conv2d_data_pack")
    else:
        data_vec = data_pad

    kernel_vec = kernel

    oshape = (batch_size, num_filter // sch.oc_bn, out_height, out_width, sch.oc_bn)
    ic = tvm.reduce_axis((0, in_channel), name='ic')

    import re
    unpack_channel_block = re.findall(r'\d+', sch.layout_out)
    if len(unpack_channel_block) == 0:
        conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
        tvm.sum(data_vec[n, ic // sch.ic_bn, oh * HSTR, ow * WSTR, ic % sch.ic_bn] *
                kernel_vec[oc_chunk, ic // sch.ic_bn, ic % sch.ic_bn, oc_block, 0, 0],
                axis=[ic]), name='conv2d') # tag='conv2d_nChwc')
        unpack_shape = (batch_size, num_filter, out_height, out_width)
        unpack = tvm.compute(unpack_shape,
                             lambda n, c, h, w: conv[n, c // sch.oc_bn, h, w, c % sch.oc_bn],
                             name='output_unpack',
                             tag='conv2d_nChwc_unpack')
    else:
        assert len(unpack_channel_block) == 1
        unpack_channel_block = int(unpack_channel_block[0])
        if unpack_channel_block == sch.oc_bn:
            return tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                    tvm.sum(data_vec[n, ic // sch.ic_bn, oh * HSTR, ow * WSTR, ic % sch.ic_bn] *
                    kernel_vec[oc_chunk, ic // sch.ic_bn, ic % sch.ic_bn, oc_block, 0, 0],
                    axis=[ic]), name='conv2d', tag='conv2d_nChwc')
        else:
            conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                    tvm.sum(data_vec[n, ic // sch.ic_bn, oh * HSTR, ow * WSTR, ic % sch.ic_bn] *
                    kernel_vec[oc_chunk, ic // sch.ic_bn, ic % sch.ic_bn, oc_block, 0, 0],
                    axis=[ic]), name='conv2d')  # tag='conv2d_nChwc')
            unpack_shape = (batch_size, num_filter // unpack_channel_block, out_height, out_width, unpack_channel_block)
            unpack = tvm.compute(unpack_shape,
                                 lambda n, C, h, w, c: conv[n, (C * unpack_channel_block + c) // sch.oc_bn, h, w, (
                                         C * unpack_channel_block + c) % sch.oc_bn],
                                 name='output_unpack',
                                 tag='conv2d_nChwc_unpack')

    return unpack


def _schedule_conv(s, wkl, data, data_pad, data_vec, kernel, conv_out, output, last):
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    DOPAD = (HPAD != 0 and WPAD != 0)

    # A, W = data, kernel_pack
    A0, A1 = data_pad, data_vec
    # schedule data
    if DOPAD and "conv2d_data_pack" in s[A1].op.tag:
        s[A0].compute_inline()
    if isinstance(s[A1].op, tvm.tensor.ComputeOp): # and  "conv2d_data_pack" in s[A1].op.tag:
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

    parallel_axis = s[C].fuse(oc_chunk, oh_outer)
    s[CC].compute_at(s[C], parallel_axis)
    if C == O:
        s[C].parallel(parallel_axis)

    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=sch.ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].fuse(oc_chunk, oh_outer)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if O0 != O:
        s[O0].compute_inline()

    if C != O:
        if len(s[O].op.axis) == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            oh_outer, oh_inner = s[O].split(oh, factor=sch.oh_factor)
            ow_outer, ow_inner = s[O].split(ow, factor=sch.ow_factor)
            s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

            parallel_axis = s[O].fuse(oc_chunk, oh_outer)
            s[C].compute_at(s[O], parallel_axis)

            _, oc_block = s[O].split(oc_block, factor=sch.oc_bn)
            s[O].vectorize(oc_block)

            s[O].parallel(parallel_axis)
        else:
            assert len(s[O].op.axis) == 4
            batch, oc, oh, ow = s[O].op.axis
            oc_chunk, oc_block = s[O].split(oc, factor=sch.oc_bn)
            oh_outer, oh_inner = s[O].split(oh, factor=sch.oh_factor)
            ow_outer, ow_inner = s[O].split(ow, factor=sch.ow_factor)
            s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

            parallel_axis = s[O].fuse(oc_chunk, oh_outer)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)

            s[O].parallel(parallel_axis)

    return s
