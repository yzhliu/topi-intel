from __future__ import absolute_import as _abs

from . import avx512_conv_common, avx512_conv_1x1

from .avx512_conv_common import AVX512ConvCommonFwd
from .avx512_conv_1x1 import AVX512Conv1x1Fwd

import tvm

from topi.nn.conv2d import conv2d, _get_schedule
from topi.nn.conv2d import _WORKLOADS, Workload
from topi.nn.conv2d import _get_workload
from topi import generic
from topi.nn.util import infer_pad, infer_stride
from topi import tag

# _SCHEDULES = [
#     # resnet 18
#     AVX512ConvCommonFwd(3, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 14, False),
#     AVX512ConvCommonFwd(16, 16, 14, False),
#     AVX512ConvCommonFwd(16, 16, 14, True),
#     AVX512ConvCommonFwd(16, 32, 7, True),
#     AVX512ConvCommonFwd(16, 32, 7, False),
#     AVX512ConvCommonFwd(16, 16, 7, True),
#     # float32 mobilenet
#     # TODO: mocked ones, need to search for best performance
#     AVX512ConvCommonFwd(3, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 28, False),
#     AVX512ConvCommonFwd(16, 16, 14, False),
#     AVX512ConvCommonFwd(16, 16, 14, False),
#     AVX512ConvCommonFwd(16, 16, 14, True),
#     AVX512ConvCommonFwd(16, 32, 7, True),
#     # resnet 50
#     AVX512ConvCommonFwd(16, 16, 28, True),
#     AVX512ConvCommonFwd(16, 16, 28, True),
#     AVX512ConvCommonFwd(16, 16, 28, True),
#     AVX512ConvCommonFwd(16, 16, 28, True),
#     AVX512ConvCommonFwd(16, 16, 28, True),
#     AVX512ConvCommonFwd(16, 16, 28, True),
#     AVX512ConvCommonFwd(16, 16, 14, True),
#     AVX512ConvCommonFwd(16, 16, 14, True),
#     AVX512ConvCommonFwd(16, 16, 14, True),
#     AVX512ConvCommonFwd(16, 16, 14, True),
#     AVX512ConvCommonFwd(16, 16, 7, True),
#     AVX512ConvCommonFwd(16, 16, 7, True),
#     AVX512ConvCommonFwd(16, 16, 7, True),
#     AVX512ConvCommonFwd(16, 16, 7, True),
# ]

_SCHEDULES = [
    # resnet 18
    AVX512ConvCommonFwd(3, 16, 28, False),
    AVX512ConvCommonFwd(16, 16, 28, False),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512ConvCommonFwd(16, 16, 28, False),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512ConvCommonFwd(16, 16, 28, False),
    AVX512ConvCommonFwd(16, 16, 14, False),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512ConvCommonFwd(16, 16, 14, True),
    AVX512ConvCommonFwd(16, 32, 7, True),
    AVX512Conv1x1Fwd(16, 16, 1, 7),
    AVX512ConvCommonFwd(16, 16, 7, True),
    # float32 mobilenet
    AVX512ConvCommonFwd(3, 16, 28, False),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 1, 7),
    AVX512Conv1x1Fwd(16, 16, 1, 7),
    # resnet 50
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 1, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 1, 7),
    AVX512Conv1x1Fwd(16, 16, 1, 7),
    AVX512Conv1x1Fwd(16, 16, 1, 7),
    AVX512Conv1x1Fwd(16, 16, 1, 7),
]

_SCHEDULES_AVX2 = [
    # resnet 18
    AVX512ConvCommonFwd(3, 8, 28, False),
    AVX512ConvCommonFwd(16, 8, 28, False),
    AVX512Conv1x1Fwd(16, 8, 1, 28),
    AVX512ConvCommonFwd(16, 8, 28, False),
    AVX512Conv1x1Fwd(16, 8, 1, 28),
    AVX512ConvCommonFwd(16, 8, 28, False),
    AVX512ConvCommonFwd(16, 8, 14, False),
    AVX512Conv1x1Fwd(16, 8, 2, 14),
    AVX512ConvCommonFwd(16, 8, 14, True),
    AVX512ConvCommonFwd(16, 8, 7, True),
    AVX512Conv1x1Fwd(16, 8, 1, 7),
    AVX512ConvCommonFwd(16, 8, 7, True),
    # float32 mobilenet
    # TODO: mocked ones, need to search for best performance
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
    # resnet 50
    AVX512Conv1x1Fwd(16, 8, 1, 28),
    AVX512Conv1x1Fwd(16, 8, 1, 28),
    AVX512Conv1x1Fwd(16, 8, 1, 28),
    AVX512Conv1x1Fwd(16, 8, 1, 28),
    AVX512Conv1x1Fwd(16, 8, 1, 28),
    AVX512Conv1x1Fwd(16, 8, 1, 28),
    AVX512Conv1x1Fwd(16, 8, 2, 14),
    AVX512Conv1x1Fwd(16, 8, 2, 14),
    AVX512Conv1x1Fwd(16, 8, 2, 14),
    AVX512Conv1x1Fwd(16, 8, 2, 14),
    AVX512Conv1x1Fwd(16, 8, 1, 7),
    AVX512Conv1x1Fwd(16, 8, 1, 7),
    AVX512Conv1x1Fwd(16, 8, 1, 7),
    AVX512Conv1x1Fwd(16, 8, 1, 7),
]

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
    workloads = _WORKLOADS + [
        # resnet 50 workloads
        Workload(in_dtype='float32', out_dtype='float32', height=56, width=56, in_filter=64, out_filter=256,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=1, wstride=1),
        Workload(in_dtype='float32', out_dtype='float32', height=56, width=56, in_filter=256, out_filter=64,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=1, wstride=1),
        Workload(in_dtype='float32', out_dtype='float32', height=56, width=56, in_filter=256, out_filter=128,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=2, wstride=2),
        Workload(in_dtype='float32', out_dtype='float32', height=28, width=28, in_filter=128, out_filter=512,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=1, wstride=1),
        Workload(in_dtype='float32', out_dtype='float32', height=56, width=56, in_filter=256, out_filter=512,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=2, wstride=2),
        Workload(in_dtype='float32', out_dtype='float32', height=28, width=28, in_filter=512, out_filter=128,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=1, wstride=1),
        Workload(in_dtype='float32', out_dtype='float32', height=28, width=28, in_filter=512, out_filter=256,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=2, wstride=2),
        Workload(in_dtype='float32', out_dtype='float32', height=14, width=14, in_filter=256, out_filter=1024,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=1, wstride=1),
        Workload(in_dtype='float32', out_dtype='float32', height=28, width=28, in_filter=512, out_filter=1024,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=2, wstride=2),
        Workload(in_dtype='float32', out_dtype='float32', height=14, width=14, in_filter=1024, out_filter=256,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=1, wstride=1),
        Workload(in_dtype='float32', out_dtype='float32', height=14, width=14, in_filter=1024, out_filter=512,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=2, wstride=2),
        Workload(in_dtype='float32', out_dtype='float32', height=7, width=7, in_filter=512, out_filter=2048,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=1, wstride=1),
        Workload(in_dtype='float32', out_dtype='float32', height=14, width=14, in_filter=1024, out_filter=2048,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=2, wstride=2),
        Workload(in_dtype='float32', out_dtype='float32', height=7, width=7, in_filter=2048, out_filter=512,
                 hkernel=1, wkernel=1, hpad=0, wpad=0, hstride=1, wstride=1)
    ]
    if wkl not in workloads:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = workloads.index(wkl)
    sch = _SCHEDULES[idx]
    return sch


@conv2d.register("cpu", override=True)
def _declaration_conv(data, kernel, stride, padding, layout, out_dtype):
    assert layout == 'NCHW', "only support NCHW convolution on rasp"
    assert data.shape[0].value == 1, "only support batch size=1 convolution on rasp"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)
    return _SCH_TO_DECL_FUNC[type(sch)](data, kernel, stride, padding, layout, out_dtype)


@generic.schedule_conv2d_nchw.register(["cpu"], override=True)
def schedule_conv2d(outs):
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
            # print('Run in x86-rasp schedule')
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            padding = infer_pad(data, data_pad)
            if data_pad is None:
                stride = infer_stride(data, kernel, output)
            else:
                stride = infer_stride(data_pad, kernel, output)

            wkl = _get_workload(data, kernel, stride, padding, output.dtype)
            sch = _get_schedule(wkl)
            return _SCH_TO_SCH_FUNC[type(sch)](s, data, data_pad, data_vec,
                                               kernel, kernel_vec, conv_out, output, outs[0])

    traverse(outs[0].op)
    return s
