from __future__ import absolute_import as _abs

from . import avx512_conv_common, avx512_conv_1x1

from .avx512_conv_common import AVX512ConvCommonFwd
from .avx512_conv_1x1 import AVX512Conv1x1Fwd

import nnvm
import nnvm.symbol as sym
from nnvm.top import registry as reg

import tvm
from topi.nn.conv2d import conv2d, _get_schedule
from topi.util import get_const_tuple, get_const_int
from topi.nn.conv2d import conv2d_nChwc
from topi.nn.conv2d import _WORKLOADS, Workload
from topi.nn.conv2d import _get_workload
from topi import generic
from topi.nn.util import infer_pad, infer_stride
from topi import tag

fp32_vec_len = 16
"""
_SCHEDULES = [
    # SSD Resnet50
    AVX512ConvCommonFwd(ic_bn=3, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=1, ow_factor=8),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=2, ow_factor=4),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=2, ow_factor=16),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=2, ow_factor=4),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=1, ow_factor=8),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=2, ow_factor=4),
    AVX512Conv1x1Fwd(ic_bn=128, oc_bn=32, oh_factor=2, ow_factor=16),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=64, oc_bn=32, oh_factor=2, ow_factor=2),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=2, ow_factor=16),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=2, ow_factor=4),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=1, ow_factor=8),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=256, oc_bn=32, oh_factor=1, ow_factor=8),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=32, oh_factor=2, ow_factor=4),
    AVX512Conv1x1Fwd(ic_bn=512, oc_bn=32, oh_factor=2, ow_factor=4),
    AVX512Conv1x1Fwd(ic_bn=512, oc_bn=32, oh_factor=1, ow_factor=8),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=32, reg_n=8, unroll_kw=True),
    # SSD Resnet50 other
    # Layer 2
    AVX512Conv1x1Fwd(ic_bn=32, oc_bn=64, oh_factor=2, ow_factor=2),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=64, reg_n=4, unroll_kw=True),
    # Layer 3
    AVX512Conv1x1Fwd(ic_bn=64, oc_bn=16, oh_factor=1, ow_factor=8),
    AVX512ConvCommonFwd(ic_bn=16, oc_bn=64, reg_n=4, unroll_kw=True),
    # Layer 4
    AVX512Conv1x1Fwd(ic_bn=64, oc_bn=4, oh_factor=2, ow_factor=4),
    AVX512ConvCommonFwd(ic_bn=4, oc_bn=128, reg_n=2, unroll_kw=False),
    # Layer 5
    AVX512Conv1x1Fwd(ic_bn=128, oc_bn=128, oh_factor=2, ow_factor=2),
    AVX512ConvCommonFwd(ic_bn=128, oc_bn=128, reg_n=1, unroll_kw=True),
    # loc_preds
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=16, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=12, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=12, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=6, reg_n=4, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=128, oc_bn=8, reg_n=2, unroll_kw=False),
    AVX512ConvCommonFwd(ic_bn=128, oc_bn=16, reg_n=1, unroll_kw=True),
    # cls_preds
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=14, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=32, oc_bn=14, reg_n=16, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=14, reg_n=8, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=64, oc_bn=14, reg_n=4, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=128, oc_bn=12, reg_n=2, unroll_kw=False),
    AVX512ConvCommonFwd(ic_bn=128, oc_bn=4, reg_n=1, unroll_kw=False),
]
"""

_SCHEDULES = [
    # SSD Resnet50
    AVX512ConvCommonFwd(ic_bn=3, oc_bn=fp32_vec_len, reg_n=32, unroll_kw=False), #0
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #1
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=32, unroll_kw=False), #2
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #3
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #4
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #5
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=32, unroll_kw=False), #6
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #7
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #8
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #9
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=32, unroll_kw=False), #10
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #11
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=32, unroll_kw=False), #12
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #13
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #14
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #15
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=32, unroll_kw=False), #16
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=1, ow_factor=32), #17
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=16, unroll_kw=False), #18
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=16), #19
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=16), #20
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=16), #21
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=16, unroll_kw=False), #22
    # SSD Resnet50 other
    # Layer 2
    # AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=16),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=8, unroll_kw=False),
    # # Layer 3
    # AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=4, ow_factor=8),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=4, unroll_kw=False),
    # # Layer 4
    # AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=4, ow_factor=4),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=2, unroll_kw=False),
    # # Layer 5
    # AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=2),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=1, unroll_kw=True),
    # # loc_preds
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=32, unroll_kw=True),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=16, unroll_kw=True),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=8, reg_n=8, unroll_kw=True),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=8, reg_n=4, unroll_kw=True),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=2, unroll_kw=False),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=1, unroll_kw=False),
    # # cls_preds
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=14, reg_n=16, unroll_kw=False),
    # AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=14, reg_n=16, unroll_kw=False),
    # AVX512ConvCommonFwd(ic_bn=64, oc_bn=14, reg_n=8, unroll_kw=True),
    # AVX512ConvCommonFwd(ic_bn=64, oc_bn=14, reg_n=4, unroll_kw=True),
    # AVX512ConvCommonFwd(ic_bn=128, oc_bn=12, reg_n=2, unroll_kw=False),
    # AVX512ConvCommonFwd(ic_bn=128, oc_bn=17, reg_n=1, unroll_kw=False),
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
    workloads = [
        # SSD Resnet50_v2 0-22
        Workload('float32', 'float32', 512, 512, 3, 64, 7, 7, 3, 3, 2, 2),
        Workload('float32', 'float32', 128, 128, 64, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 128, 128, 64, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 128, 128, 64, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 128, 128, 256, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 128, 128, 256, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 128, 128, 128, 128, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 64, 64, 128, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 128, 128, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 64, 64, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 64, 64, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 64, 64, 512, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 64, 64, 256, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 32, 32, 256, 1024, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 64, 64, 512, 1024, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 32, 32, 1024, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 32, 32, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 32, 32, 1024, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 32, 32, 512, 512, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 16, 16, 512, 2048, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 32, 32, 1024, 2048, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 16, 16, 2048, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 16, 16, 512, 512, 3, 3, 1, 1, 1, 1),
        # SSD Resnet50_v2 others 23-42
        # layer2
        Workload('float32', 'float32', 16, 16, 2048, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 16, 16, 256, 512, 3, 3, 1, 1, 2, 2),
        # layer3
        Workload('float32', 'float32', 8, 8, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 8, 8, 128, 256, 3, 3, 1, 1, 2, 2),
        # layer4
        Workload('float32', 'float32', 4, 4, 256, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 4, 4, 128, 256, 3, 3, 1, 1, 2, 2),
        # layer5
        Workload('float32', 'float32', 2, 2, 256, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 2, 2, 128, 128, 3, 3, 1, 1, 2, 2),
        # loc_preds
        Workload('float32', 'float32', 32, 32, 1024, 16, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 16, 16, 2048, 24, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 8, 8, 512, 24, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 4, 4, 256, 24, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 2, 2, 256, 16, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 1, 1, 128, 16, 3, 3, 1, 1, 1, 1),
        # cls_preds
        Workload('float32', 'float32', 32, 32, 1024, 84, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 16, 16, 2048, 126, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 8, 8, 512, 126, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 4, 4, 256, 126, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 2, 2, 256, 84, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 1, 1, 128, 84, 3, 3, 1, 1, 1, 1),
    ]
    if wkl not in workloads:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = workloads.index(wkl)
    sch = _SCHEDULES[idx]
    return sch


@reg.register_alter_op_layout("conv2d")
def weight_prepack_conv2d(attrs, inputs, tinfos):
    copy_inputs = [inputs[i] for i in range(len(inputs))]

    data = tinfos[0]
    kernel = tinfos[1]

    import ast
    padding = ast.literal_eval(attrs['padding'])
    stride = ast.literal_eval(attrs['strides'])

    wkl = _get_workload(data, kernel, stride, padding, 'float32')
    sch = _get_schedule_conv(wkl)
    is_kernel_1x1 = isinstance(sch, AVX512Conv1x1Fwd)
    ic_bn, oc_bn = sch.ic_bn, sch.oc_bn

    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['layout'] = 'NCHW%dc' % ic_bn
    new_attrs['out_layout'] = 'NCHW%dc' % oc_bn

    if is_kernel_1x1:
        # (oc, ic, h, w) -> (OC, IC, ic, oc, h, w)
        new_attrs['kernel_layout'] = 'OI%di%doHW' % (ic_bn, oc_bn)
    else:
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)

    return sym.conv2d_nChwc(*copy_inputs, **new_attrs)


@conv2d_nChwc.register("cpu", override=True)
def _declaration_conv(data, kernel, num_filter, kernel_size, stride, padding, out_dtype):
    assert data.shape[0].value == 1, "only support batch size=1 convolution on avx"
    n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
    ic = ic_chunk * ic_block
    oc = num_filter
    kh, kw = kernel_size
    wkl = _get_workload(tvm.placeholder((n, ic, h, w), dtype=out_dtype),
                        tvm.placeholder((oc, ic, kh, kw), dtype=out_dtype), stride, padding, out_dtype)
    sch = _get_schedule(wkl)
    return _SCH_TO_DECL_FUNC[type(sch)](data, kernel, stride, padding, out_dtype)


@generic.schedule_conv2d_nChwc.register(["cpu"], override=True)
def schedule_conv2d_nChwc(num_filter, kernel_size, outs):
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

        if 'conv2d_nChwc' in op.tag:
            output = op.output(0)
            # conv_out = op.input_tensors[0]
            conv_out = output
            kernel = conv_out.op.input_tensors[1]
            # kernel = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, tvm.tensor.ComputeOp) and len(data_vec.op.input_tensors) > 0 and "pad" not in data_vec.op.tag \
                else data_vec
            data_pad = None

            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
            ic = ic_chunk * ic_block
            original_data = tvm.placeholder((n, ic, h, w), dtype=output.dtype)

            if data_pad is not None:
                n, _, pad_h, pad_w, _ = [x.value for x in data_pad.shape]
                original_data_pad = tvm.placeholder((n, ic, pad_h, pad_w), dtype=output.dtype)
                padding = infer_pad(original_data, original_data_pad)
            else:
                padding = (0, 0)

            oc = num_filter
            kh, kw = kernel_size
            original_kernel = tvm.placeholder((oc, ic, kh, kw), dtype=output.dtype)

            n, oc_chunk, oh, ow, oc_block = [x.value for x in output.shape]
            original_output = tvm.placeholder((n, oc_chunk*oc_block, oh, ow), dtype=output.dtype)

            if data_pad is None:
                stride = infer_stride(original_data, original_kernel, original_output)
            else:
                stride = infer_stride(original_data_pad, original_kernel, original_output)

            wkl = _get_workload(original_data, original_kernel, stride, padding, output.dtype)
            sch = _get_schedule(wkl)
            _SCH_TO_SCH_FUNC[type(sch)](s, data, data_pad, data_vec,
                                        kernel, conv_out, output, outs[0])


    traverse(outs[0].op)
    return s
