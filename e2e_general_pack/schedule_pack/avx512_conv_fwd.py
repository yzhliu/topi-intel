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
from topi.nn.conv2d import conv2d_NCHWc
from topi.nn.conv2d import _WORKLOADS, Workload
from topi.nn.conv2d import _get_workload
from topi import generic
from topi.nn.util import infer_pad, infer_stride
from topi import tag

fp32_vec_len = 16
_SCHEDULES = [
    # workloads of resnet18_v1 on imagenet
    AVX512ConvCommonFwd(3, fp32_vec_len, 28, False),
    AVX512ConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
    AVX512Conv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 28),
    AVX512ConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
    AVX512Conv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 28),
    AVX512ConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
    AVX512ConvCommonFwd(fp32_vec_len, fp32_vec_len, 14, False),
    AVX512Conv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
    AVX512ConvCommonFwd(fp32_vec_len, fp32_vec_len, 14, True),
    # AVX512ConvCommonFwd(16, 32, 7, True),
    AVX512ConvCommonFwd(fp32_vec_len, fp32_vec_len, 7, True),
    AVX512Conv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 7),
    AVX512ConvCommonFwd(fp32_vec_len, fp32_vec_len, 7, True),
    # workloads of resnet34_v1 on imagenet, no extra workload required
    # workloads of resnet50_v1 on imagenet
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=14),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=14),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=14),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=14),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=7),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=7),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=7),
    AVX512Conv1x1Fwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, oh_factor=2, ow_factor=7),
    # workloads of resnet101_v1 on imagenet, no extra workload required
    # workloads of resnet152_v1 on imagenet, no extra workload required
    # workloads of resnet18_v2 on imagenet, no extra workload required
    # workloads of resnet34_v2 on imagenet, no extra workload required
    # workloads of resnet50_v2 on imagenet
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=28, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=14, unroll_kw=False),
    AVX512ConvCommonFwd(ic_bn=fp32_vec_len, oc_bn=fp32_vec_len, reg_n=7, unroll_kw=True),
    # workloads of resnet101_v2 on imagenet, no extra workload required
    # workloads of resnet152_v2 on imagenet, no extra workload required
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
        # workloads of resnet18_v1 on imagenet 12 0-11
        Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
        # workloads of resnet34_v1 on imagenet, no extra workload required
        # workloads of resnet50_v1 on imagenet 14 12-25
        Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),
        # workloads of resnet101_v1 on imagenet, no extra workload required
        # workloads of resnet152_v1 on imagenet, no extra workload required
        # workloads of resnet18_v2 on imagenet, no extra workload required
        # workloads of resnet34_v2 on imagenet, no extra workload required
        # workloads of resnet50_v2 on imagenet 3 26-28
        Workload('float32', 'float32', 56, 56, 128, 128, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 28, 28, 256, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 14, 14, 512, 512, 3, 3, 1, 1, 2, 2),
        # workloads of resnet101_v2 on imagenet, no extra workload required
        # workloads of resnet152_v2 on imagenet, no extra workload required
    ]
    if wkl not in workloads:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = workloads.index(wkl)
    sch = _SCHEDULES[idx]
    return sch


@reg.register_alter_op_layout("conv2d")
def alter_conv2d_layout(attrs, inputs, tinfos):
    copy_inputs = [s for s in inputs]

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

    return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)


@conv2d_NCHWc.register("cpu", override=True)
def _declaration_conv(data, kernel, num_filter, kernel_size, stride, padding, out_dtype):
    assert data.shape[0].value == 1, "only support batch size=1 convolution on avx"
    n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
    ic = ic_chunk * ic_block
    oc = num_filter
    kh, kw = kernel_size
    wkl = _get_workload(tvm.placeholder((n, ic, h, w), dtype=out_dtype),
                        tvm.placeholder((oc, ic, kh, kw), dtype=out_dtype), stride, padding, out_dtype)
    sch = _get_schedule(wkl)
    return _SCH_TO_DECL_FUNC[type(sch)](wkl, data, kernel)


@generic.schedule_conv2d_NCHWc.register(["cpu"], override=True)
def schedule_conv2d_NCHWc(num_filter, kernel_size, stride, padding, outs):
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

            oc = num_filter
            kh, kw = kernel_size
            original_kernel = tvm.placeholder((oc, ic, kh, kw), dtype=output.dtype)

            wkl = _get_workload(original_data, original_kernel, stride, padding, output.dtype)
            sch = _get_schedule(wkl)
            _SCH_TO_SCH_FUNC[type(sch)](s, wkl, data, data_pad, data_vec,
                                        kernel, conv_out, output, outs[0])


    traverse(outs[0].op)
    return s
