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
from topi.nn.conv2d_prepack import conv2d_nChwc
from topi.nn.conv2d import _WORKLOADS, Workload
from topi.nn.conv2d import _get_workload
from topi import generic
from topi.nn.util import infer_pad, infer_stride
from topi import tag

fp32_vec_len = 16
_SCHEDULES = [
    # workloads of resnet18_v1 on imagenet
    AVX512ConvCommonFwd(3, fp32_vec_len, 28, False),
    AVX512ConvCommonFwd(16, fp32_vec_len, 28, False),
    AVX512Conv1x1Fwd(16, fp32_vec_len, 1, 28),
    AVX512ConvCommonFwd(16, fp32_vec_len, 28, False),
    AVX512Conv1x1Fwd(16, fp32_vec_len, 1, 28),
    AVX512ConvCommonFwd(16, fp32_vec_len, 28, False),
    AVX512ConvCommonFwd(16, fp32_vec_len, 14, False),
    AVX512Conv1x1Fwd(16, fp32_vec_len, 2, 14),
    AVX512ConvCommonFwd(16, fp32_vec_len, 14, True),
    AVX512ConvCommonFwd(16, fp32_vec_len, 7, True),
    AVX512Conv1x1Fwd(16, fp32_vec_len, 1, 7),
    AVX512ConvCommonFwd(16, fp32_vec_len, 7, True),
    # workloads of resnet34_v1 on imagenet, no extra workload required
    # workloads of resnet50_v1 on imagenet
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=28),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=14),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=14),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=14),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=14),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=7),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=7),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=7),
    AVX512Conv1x1Fwd(ic_bn=16, oc_bn=16, oh_factor=2, ow_factor=7),
    # workloads of resnet101_v1 on imagenet, no extra workload required
    # workloads of resnet152_v1 on imagenet, no extra workload required
    # workloads of resnet18_v2 on imagenet, no extra workload required
    # workloads of resnet34_v2 on imagenet, no extra workload required
    # workloads of resnet50_v2 on imagenet
    AVX512ConvCommonFwd(ic_bn=16, oc_bn=16, reg_n=28, unroll_kw=True),
    AVX512ConvCommonFwd(ic_bn=16, oc_bn=16, reg_n=14, unroll_kw=False),
    AVX512ConvCommonFwd(ic_bn=16, oc_bn=16, reg_n=7, unroll_kw=True),
    # workloads of resnet101_v2 on imagenet, no extra workload required
    # workloads of resnet152_v2 on imagenet, no extra workload required
    # workloads of mobilenet 1.0 on imagenet 10 29-38
    AVX512ConvCommonFwd(3, 16, 28, True),
    AVX512Conv1x1Fwd(16, 32, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 32, 2, 28),
    AVX512Conv1x1Fwd(16, 32, 2, 14),
    AVX512Conv1x1Fwd(16, 32, 2, 14),
    AVX512Conv1x1Fwd(16, 32, 2, 7),
    AVX512Conv1x1Fwd(16, 16, 2, 7),
    # workloads of mobilenet 0.75 on imagenet 9 39-47
    AVX512ConvCommonFwd(3, 16, 28, False),
    AVX512Conv1x1Fwd(16, 32, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 32, 2, 14),
    AVX512Conv1x1Fwd(16, 32, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 2, 7),
    AVX512Conv1x1Fwd(16, 16, 2, 7),
    # workloads of mobilenet 0.5 on imagenet 8 48-55
    AVX512ConvCommonFwd(3, 32, 28, True),
    AVX512Conv1x1Fwd(16, 32, 2, 28),
    AVX512Conv1x1Fwd(16, 32, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 2, 7),
    AVX512Conv1x1Fwd(16, 16, 2, 7),
    # workloads of mobilenet 0.25 on imagenet 56-65
    AVX512ConvCommonFwd(3, 16, 28, False),
    AVX512Conv1x1Fwd(8, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 32, 2, 28),
    AVX512Conv1x1Fwd(16, 32, 2, 28),
    AVX512Conv1x1Fwd(16, 32, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 28),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 16, 2, 14),
    AVX512Conv1x1Fwd(16, 32, 2, 7),
    AVX512Conv1x1Fwd(16, 16, 2, 7),
    # workloads of vgg11 on imagenet 66-72
    AVX512ConvCommonFwd(3, 32, 32, True),
    AVX512ConvCommonFwd(16, 16, 28, True),
    AVX512ConvCommonFwd(16, 32, 28, False),
    AVX512ConvCommonFwd(16, 16, 28, True),
    AVX512ConvCommonFwd(16, 32, 28, False),
    AVX512ConvCommonFwd(16, 16, 28, True),
    AVX512ConvCommonFwd(16, 16, 14, False),
    # workloads of vgg13 on imagenet 73-74
    AVX512ConvCommonFwd(16, 16, 32, True),
    AVX512ConvCommonFwd(16, 16, 28, True),
    # workloads of vgg16 on imagenet, no extra workload required
    # workloads of vgg19 on imagenet, no extra workload required
    # workloads of vgg11_bn on imagenet, no extra workload required
    # workloads of vgg13_bn on imagenet, no extra workload required
    # workloads of vgg16_bn on imagenet, no extra workload required
    # workloads of vgg19_bn on imagenet, no extra workload required
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
        # workloads of mobilenet 1.0 on imagenet 10 29-38
        Workload('float32', 'float32', 224, 224, 3, 32, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 112, 112, 32, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 512, 1024, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1),
        # workloads of mobilenet 0.75 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 24, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 112, 112, 24, 48, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 48, 96, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 96, 96, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 96, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 192, 384, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 384, 384, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 384, 768, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 768, 768, 1, 1, 0, 0, 1, 1),
        # workloads of mobilenet 0.5 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 16, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 112, 112, 16, 32, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 32, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 64, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 128, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 256, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 512, 512, 1, 1, 0, 0, 1, 1),
        # workloads of mobilenet 0.25 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 8, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 112, 112, 8, 16, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 16, 32, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 32, 32, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 32, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 64, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 64, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 128, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 128, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 256, 256, 1, 1, 0, 0, 1, 1),
        # workloads of vgg11 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 112, 112, 64, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 128, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 256, 512, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 512, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 512, 512, 3, 3, 1, 1, 1, 1),
        # workloads of vgg13 on imagenet
        Workload('float32', 'float32', 224, 224, 64, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 112, 112, 128, 128, 3, 3, 1, 1, 1, 1),
        # workloads of vgg16 on imagenet, no extra workload required
        # workloads of vgg19 on imagenet, no extra workload required
        # workloads of vgg11_bn on imagenet, no extra workload required
        # workloads of vgg13_bn on imagenet, no extra workload required
        # workloads of vgg16_bn on imagenet, no extra workload required
        # workloads of vgg19_bn on imagenet, no extra workload required
        # workloads of densenet 121 on imagenet
        Workload('float32', 'float32', 56, 56, 128, 32, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 96, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 160, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 192, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 224, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 32, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 160, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 192, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 224, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 256, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 288, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 320, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 352, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 384, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 416, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 448, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 480, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 128, 32, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 288, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 320, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 352, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 384, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 416, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 448, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 480, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 544, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 576, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 608, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 640, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 672, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 704, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 736, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 768, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 800, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 832, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 864, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 896, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 928, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 960, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 992, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 128, 32, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 7, 7, 544, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 576, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 608, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 640, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 672, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 704, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 736, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 768, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 800, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 832, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 864, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 896, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 928, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 960, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 992, 128, 1, 1, 0, 0, 1, 1),
        # workloads of densenet 161 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 96, 7, 7, 3, 3, 2, 2),
        Workload('float32', 'float32', 56, 56, 96, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 192, 48, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 144, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 192, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 240, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 288, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 336, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 384, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 192, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 192, 48, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 240, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 288, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 336, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 384, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 432, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 480, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 528, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 576, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 624, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 672, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 720, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 768, 384, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 384, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 192, 48, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 432, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 480, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 528, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 576, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 624, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 672, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 720, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 768, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 816, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 864, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 912, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 960, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1008, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1056, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1104, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1152, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1200, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1248, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1296, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1344, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1392, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1440, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1488, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1536, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1584, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1632, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1680, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1728, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1776, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1824, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1872, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1920, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1968, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 2016, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 2064, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 2112, 1056, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1056, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 192, 48, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 7, 7, 1104, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1152, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1200, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1248, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1296, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1344, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1392, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1440, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1488, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1536, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1584, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1632, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1680, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1728, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1776, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1824, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1872, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1920, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1968, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 2016, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 2064, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 2112, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 2160, 192, 1, 1, 0, 0, 1, 1),
        # workloads of densenet 169 on imagenet
        Workload('float32', 'float32', 14, 14, 1024, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1056, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1088, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1120, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1152, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1184, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1216, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1248, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1280, 640, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1024, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1056, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1088, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1120, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1152, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1184, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1216, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1248, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1280, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1312, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1344, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1376, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1408, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1440, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1472, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1504, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1536, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1568, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1600, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1632, 128, 1, 1, 0, 0, 1, 1),
        # workloads of densenet 201 on imagenet
        Workload('float32', 'float32', 14, 14, 1024, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1056, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1088, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1120, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1152, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1184, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1216, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1248, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1280, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1312, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1344, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1376, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1408, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1440, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1472, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1504, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1536, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1568, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1600, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1632, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1664, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1696, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1728, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1760, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1792, 896, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1024, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1056, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1088, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1120, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1152, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1184, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1216, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1248, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1280, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1312, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1344, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1376, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1408, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1440, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1472, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1504, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1536, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1568, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1600, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1632, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1664, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1696, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1728, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1760, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1792, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1824, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1856, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1888, 128, 1, 1, 0, 0, 1, 1),
        # workloads of alexnet 201 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 64, 11, 11, 2, 2, 4, 4),
        Workload('float32', 'float32', 27, 27, 64, 192, 5, 5, 2, 2, 1, 1),
        Workload('float32', 'float32', 13, 13, 192, 384, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 13, 13, 384, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 13, 13, 256, 256, 3, 3, 1, 1, 1, 1),
        # workloads of squeezenet1.0 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 96, 7, 7, 0, 0, 2, 2),
        Workload('float32', 'float32', 54, 54, 96, 16, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 54, 54, 16, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 54, 54, 16, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 54, 54, 128, 16, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 54, 54, 128, 32, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 54, 54, 32, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 54, 54, 32, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 27, 27, 256, 32, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 27, 27, 32, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 27, 27, 32, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 27, 27, 256, 48, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 27, 27, 48, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 27, 27, 48, 192, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 27, 27, 384, 48, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 27, 27, 384, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 27, 27, 64, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 27, 27, 64, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 13, 13, 512, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 13, 13, 64, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 13, 13, 64, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 13, 13, 512, 1000, 1, 1, 0, 0, 1, 1),
        # workloads of squeezenet1.1 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 64, 3, 3, 0, 0, 2, 2),
        Workload('float32', 'float32', 55, 55, 64, 16, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 55, 55, 16, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 55, 55, 16, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 55, 55, 128, 16, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 27, 27, 128, 32, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 13, 13, 256, 48, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 13, 13, 48, 192, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 13, 13, 48, 192, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 13, 13, 384, 48, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 13, 13, 384, 64, 1, 1, 0, 0, 1, 1),
    ]
    if wkl not in workloads:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = workloads.index(wkl)
    # if idx >= len(_SCHEDULES):
    #     sch = AVX512ConvCommonFwd(16, fp32_vec_len, 28, False)
    # else:
    sch = _SCHEDULES[idx]
    return sch


@reg.register_weight_prepack("max_pool2d")
def weight_prepack_max_pool2d(attrs, inputs, tinfos):
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['layout'] = 'NCHW_c'
    return sym.max_pool2d(inputs[0], **new_attrs)


@reg.register_weight_prepack("avg_pool2d")
def weight_prepack_avg_pool2d(attrs, inputs, tinfos):
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['layout'] = 'NCHW_c'
    return sym.avg_pool2d(inputs[0], **new_attrs)


@reg.register_weight_prepack("global_avg_pool2d")
def weight_prepack_global_avg_pool2d(attrs, inputs, tinfos):
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['layout'] = 'NCHW_c'
    return sym.global_avg_pool2d(inputs[0], **new_attrs)


@reg.register_weight_prepack("conv2d")
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
