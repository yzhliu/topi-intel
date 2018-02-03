from __future__ import absolute_import as _abs

from . import avx512_conv_common, avx512_conv_1x1

from .avx512_conv_common import AVX512ConvCommonFwd
from .avx512_conv_1x1 import AVX512Conv1x1Fwd

import tvm

from topi.nn.conv2d import conv2d, _get_schedule
from topi.nn.conv2d import _WORKLOADS
from topi.nn.conv2d import _get_workload
from topi import generic
from topi.nn.util import infer_pad, infer_stride

_SCHEDULES = [
    # float32 imagenet
    AVX512ConvCommonFwd(3, 16, 28, False),
    AVX512ConvCommonFwd(16, 16, 28, False),
    AVX512ConvCommonFwd(16, 16, 28, False),
    AVX512ConvCommonFwd(16, 16, 28, False),
    AVX512ConvCommonFwd(16, 16, 28, False),
    AVX512ConvCommonFwd(16, 16, 28, False),
    AVX512ConvCommonFwd(16, 16, 14, False),
    AVX512ConvCommonFwd(16, 16, 14, False),
    AVX512ConvCommonFwd(16, 16, 14, True),
    AVX512ConvCommonFwd(16, 32, 7, True),
    AVX512ConvCommonFwd(16, 32, 7, False),
    AVX512ConvCommonFwd(16, 16, 7, True)
]

# _SCHEDULES = [
#     # float32 imagenet
#     AVX512ConvCommonFwd(3, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512Conv1x1Fwd(16, 16, 1, 28),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512Conv1x1Fwd(16, 16, 1, 28),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 14, False),
#     AVX512Conv1x1Fwd(16, 16, 2, 14),
#     AVX512ConvCommonFwd(16, 16, 14, True),
#     AVX512ConvCommonFwd(16, 32, 7, True),
#     AVX512Conv1x1Fwd(16, 16, 1, 7),
#     AVX512ConvCommonFwd(16, 16, 7, True)
# ]

_SCH_TO_DECL_FUNC = {
    AVX512ConvCommonFwd: avx512_conv_common._declaration_conv,
    AVX512Conv1x1Fwd: avx512_conv_1x1._declaration_conv
}

_SCH_TO_SCH_FUNC = {
    AVX512ConvCommonFwd: avx512_conv_common._schedule_conv,
    AVX512Conv1x1Fwd: avx512_conv_1x1._schedule_conv
}


@_get_schedule.register("cpu", override=True)
def _get_schedule_conv(wkl):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    sch = _SCHEDULES[idx]
    return sch


@conv2d.register("cpu", override=True)
def _declaration_conv(data, kernel, stride, padding, layout, out_dtype):
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)
    return _SCH_TO_DECL_FUNC[type(sch)](data, kernel, stride, padding, layout, out_dtype)


@generic.schedule_conv2d_nchw.register(["cpu"], override=True)
def schedule_conv2d(outs):
    s = tvm.create_schedule([x.op for x in outs])

    op = outs[0].op
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

    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)
    wkl = _get_workload(data, kernel, stride, padding, output.dtype)
    sch = _get_schedule(wkl)
    return _SCH_TO_SCH_FUNC[type(sch)](s, data, data_pad, data_vec, kernel, kernel_pack, conv_out, output)
