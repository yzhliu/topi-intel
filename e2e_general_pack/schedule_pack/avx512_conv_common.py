from __future__ import absolute_import as _abs
import tvm
from topi.util import get_const_tuple, get_const_int
from collections import namedtuple

from topi.nn.conv2d import _get_schedule
from topi.nn.pad import pad

AVX512ConvCommonFwd = namedtuple('AVX512ConvCommonFwd', ['ic_bn', 'oc_bn', 'reg_n', 'unroll_kw'])

def _declaration_conv(wkl, data, kernel):
    sch = _get_schedule(wkl)

    out_dtype = wkl.out_dtype
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size, in_channel_chunk, in_height, in_width, in_channel_block = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width, _, co = get_const_tuple(kernel.shape)
    num_filter *= co

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

    # pack data
    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    in_channel = in_channel_block * in_channel_chunk
    data_vec = data_pad

    kernel_vec = kernel

    # convolution
    oshape = (batch_size, num_filter//sch.oc_bn, out_height, out_width, sch.oc_bn)

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_vec[n, ic//sch.ic_bn, oh*HSTR+kh, ow*WSTR+kw, ic%sch.ic_bn].astype(out_dtype) *
                               kernel_vec[oc_chunk, ic//sch.ic_bn, kh, kw, ic%sch.ic_bn, oc_block],
                               axis=[ic, kh, kw]),
                       name='conv2d_NCHWc', tag="conv2d_NCHWc")

    return conv


def _schedule_conv(s, wkl, data, data_pad, data_vec, kernel, conv_out, output, last):
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    DOPAD = (HPAD != 0 and WPAD != 0)

    A0, A1 = data_pad, data_vec
    # schedule data
    if isinstance(s[A1].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A1].op.axis
        parallel_axis = s[A1].fuse(ic_chunk, ih)
        s[A1].parallel(parallel_axis)

    # schedule conv
    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, 'global')

    _, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=sch.reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=sch.reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    if sch.unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if O0 != O:
        s[O0].compute_inline()

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        ow_chunk, ow_block = s[O].split(ow, factor=sch.reg_n)
        s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        parallel_axis = s[O].fuse(oc_chunk, oh)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)

        s[O].parallel(parallel_axis)

    return s
