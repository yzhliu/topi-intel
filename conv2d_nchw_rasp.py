import numpy as np
import tvm
import topi
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from topi.nn.conv2d import SpatialPack, Im2ColPack, _WORKLOADS
from topi.nn.conv2d import _get_workload
from topi.nn.util import infer_pad, infer_stride
from topi import tag
from topi.nn import pad

_SCHEDULES = [
    # float32 imagenet
    SpatialPack(1, 8, 4, 1, 4, True),
    SpatialPack(1, 7, 4, 2, 4, True),
    SpatialPack(1, 4, 8, 4, 1, True),
    SpatialPack(1, 4, 4, 1, 16, False),
    SpatialPack(1, 4, 8, 4, 8, False),
    SpatialPack(1, 7, 4, 3, 8, True),
    SpatialPack(1, 2, 8, 1, 8, True),
    SpatialPack(2, 1, 16, 1, 4, True),
    SpatialPack(1, 7, 4, 1, 1, True),
    Im2ColPack(7, 4, 1, 16, True),
    Im2ColPack(7, 4, 1, 8, False),
    Im2ColPack(7, 4, 1, 16, False),

    # float32 mobilenet
    SpatialPack(2, 2, 4, 28, 1, True),
    SpatialPack(1, 4, 8, 14, 1, False),
    SpatialPack(1, 2, 16, 8, 1, True),
    SpatialPack(1, 4, 8, 8, 8, True),
    SpatialPack(2, 2, 8, 1, 1, False),
    SpatialPack(1, 4, 8, 4, 8, False),
    SpatialPack(2, 2, 8, 1, 4, False),
    SpatialPack(2, 2, 8, 1, 8, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 4, True),
]


def _schedule_conv2d(wkl):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)
    print('idx = ' + str(idx))
    sch = _SCHEDULES[idx]
    return sch


def _spatial_get_sch(data, kernel, stride, padding, out_dtype):
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _schedule_conv2d(wkl)
    return (wkl, sch)


def traverse(s, op):
    """Traverse operators from computation graph"""
    # inline all one-to-one-mapping operators except the last stage (output)
    if tag.is_broadcast(op.tag):
        if op not in s.outputs:
            s[op].compute_inline()
        for tensor in op.input_tensors:
            if tensor.op.input_tensors:
                traverse(tensor.op)


def _spatial_pack_data_only(wkl, sch, data):
    H, W = wkl.height, wkl.width
    CI, CO = wkl.in_filter, wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    HCAT, WCAT = KH-1, KW-1

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    dshape = (1, CI, H, W)
    dpshape = (1, CI, TH, TW)
    dvshape = (1, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT)

    print(dshape)
    print(dpshape)
    print(dvshape)
    print(VH*HSTR)
    print(VW*WSTR)

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw: \
        data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    s = tvm.create_schedule(data_vec.op)
    traverse(s, data_vec.op)

    # schedule for data_vec
    A0, A1 = data_pad, data_vec
    if DOPAD:
        s[A0].compute_inline()
    _, h, _, _, _, _ = s[A1].op.axis
    if sch.ba == 1:
        oaxis = h
        paxis = h
    else:
        oh, ih = s[A1].split(h, sch.ba)
        oaxis = oh
        paxis = ih
    s[A1].parallel(paxis)
    s[A1].pragma(oaxis, "parallel_launch_point")
    s[A1].pragma(paxis, "parallel_stride_pattern")
    s[A1].pragma(oaxis, "parallel_barrier_when_finish")

    return data_vec, s


def _spatial_pack_kernel_only(wkl, sch, kernel):
    H, W = wkl.height, wkl.width
    CI, CO = wkl.in_filter, wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    HCAT, WCAT = KH-1, KW-1

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    kshape = (CO, CI, KH, KW)
    kvshape = (CO//VC, CI, KH, KW, VC)

    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc: \
        kernel[co*VC+vc][ci][dh][dw], name='kernel_vec')

    s = tvm.create_schedule(kernel_vec.op)
    traverse(s, kernel_vec.op)

    B, B0 = kernel, kernel_vec
    co, _, _, _, _ = s[B0].op.axis
    if sch.bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[B0].split(co, sch.bc)
        oaxis = oco
        paxis = ico
    s[B0].parallel(paxis)
    s[B0].pragma(oaxis, "parallel_launch_point")
    s[B0].pragma(paxis, "parallel_stride_pattern")
    s[B0].pragma(oaxis, "parallel_barrier_when_finish")

    return kernel_vec, s


def _spatial_conv_only(wkl, sch, data_vec, kernel_vec, out_dtype):
    H, W = wkl.height, wkl.width
    CI, CO = wkl.in_filter, wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    HCAT, WCAT = KH - 1, KW - 1

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    TH = H + 2 * HPAD
    TW = W + 2 * WPAD
    OH = (H + 2 * HPAD - KH) // HSTR + 1
    OW = (W + 2 * WPAD - KW) // WSTR + 1

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')

    ovshape = (1, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
        tvm.sum(data_vec[n, h, w, ci, vh * HSTR + dh, vw * WSTR + dw].astype(out_dtype) *
                kernel_vec[co, ci, dh, dw, vc].astype(out_dtype),
                axis=[ci, dh, dw]), name='conv')
    output = tvm.compute(oshape, lambda n, co, h, w:
        conv[n][co // VC][h // VH][w // VW][h % VH][w % VW][co % VC],
                         name='output_unpack', tag='spatial_conv_output')

    C0, C = conv, output

    s = tvm.create_schedule(C.op)
    traverse(s, C.op)

    CC = s.cache_write(C0, "global")
    _, co, oh, ow, vh, vw, vc = s[C0].op.axis
    if UNROLL:
        s[C0].unroll(vw)
    s[C0].vectorize(vc)

    s[CC].compute_at(s[C0], ow)
    _, co, oh, ow, vh, vw, vc = s[CC].op.axis
    ci, dh, dw = s[CC].op.reduce_axis
    s[CC].reorder(ci, dh, vh, dw, vw, vc)

    if UNROLL:
        s[CC].unroll(vw)
    s[CC].vectorize(vc)

    n, co, h, w = s[C].op.axis
    co, vc = s[C].split(co, VC)
    oh, ow, vh, vw = s[C].tile(h, w, VH, VW)
    s[C].reorder(n, co, oh, ow, vh, vw, vc)
    # if C != C1:
    #     s[C1].compute_inline()
    s[C0].compute_at(s[C], ow)

    if sch.bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[C].split(co, sch.bc)
        oaxis = oco
        paxis = ico

    s[C].parallel(paxis)
    s[C].pragma(oaxis, "parallel_launch_point")
    s[C].pragma(paxis, "parallel_stride_pattern")
    s[C].pragma(oaxis, "parallel_barrier_when_finish")

    return C, s


# TODO
def _spatial_conv_unpack_only(wkl, sch, conv):
    H, W = wkl.height, wkl.width
    CI, CO = wkl.in_filter, wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    HCAT, WCAT = KH - 1, KW - 1

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    TH = H + 2 * HPAD
    TW = W + 2 * WPAD
    OH = (H + 2 * HPAD - KH) // HSTR + 1
    OW = (W + 2 * WPAD - KW) // WSTR + 1

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')

    ovshape = (1, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    output = tvm.compute(oshape, lambda n, co, h, w:
    conv[n][co // VC][h / VH][w // VW][h % VH][w % VW][co % VC],
                         name='output_unpack', tag='spatial_conv_output')

    return output


def _spatial_pack(data, kernel, stride, padding, out_dtype):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _schedule_conv2d(wkl)

    H, W = wkl.height, wkl.width
    CI, CO = wkl.in_filter, wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    HCAT, WCAT = KH-1, KW-1

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    dshape = (1, CI, H, W)
    dpshape = (1, CI, TH, TW)
    dvshape = (1, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT)

    kshape = (CO, CI, KH, KW)
    kvshape = (CO/VC, CI, KH, KW, VC)

    ovshape = (1, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw: \
        data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc: \
        kernel[co*VC+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')

    conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
        tvm.sum(data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw].astype(out_dtype) *
                kernel_vec[co, ci, dh, dw, vc].astype(out_dtype),
                axis=[ci, dh, dw]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n][co//VC][h/VH][w//VW][h%VH][w%VW][co%VC],
                         name='output_unpack', tag='spatial_conv_output')

    return output


def _im2col_pack(wkl, sch, data, kernel, stride, padding, out_dtype):
    """ Compute convolution with im2col pack layout. """
    assert data.shape[0].value == 1, "im2col pack convolution only support batch size=1"

    N = 1
    H, W = wkl.height, wkl.width
    CI = wkl.in_filter
    CO = wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.hpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    P = sch.vp
    Q = sch.vq
    UNROLL = sch.unroll

    dshape = (N, CI, H, W)
    dpshape = (N, CI, H+2*HPAD, W+2*WPAD)
    dcshape = (N, OH, OW, CI, KH, KW)
    dvshape = (N, OH * OW // P, CI, KH, KW, P)

    kshape = (CO, CI, KH, KW)
    kvshape = (CO // Q, CI, KH, KW, Q)

    ovshape = (N, CO // Q, OH * OW // P, P, Q)
    oshape = (N, CO, OH, OW)

    ############### declaration

    DO_PAD = (wkl.hpad != 0 and wkl.wpad != 0)
    if DO_PAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    data_col = tvm.compute(dcshape, lambda n, oh, ow, ci, hk, wk: \
        data_pad[n][ci][oh*HSTR+hk][ow*WSTR+wk], name='data_col')

    data_vec = tvm.compute(dvshape, lambda n, im, ci, hk, wk, vim: \
        data_col[n][(im*P+vim)//OW][(im*P+vim)%OW][ci][hk][wk], name='data_vec')


    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc: \
        kernel[co*Q+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    hk = tvm.reduce_axis((0, KH), name='hk')
    wk = tvm.reduce_axis((0, KW), name='wk')

    conv = tvm.compute(ovshape, lambda n, co, im, vim, vco: \
        tvm.sum(data_vec[n][im][ci][hk][wk][vim].astype(out_dtype) *
                kernel_vec[co][ci][hk][wk][vco].astype(out_dtype),
                axis=[ci, hk, wk]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w: \
                         conv[n][co//Q][(h*OW+w)//P][(h*OW+w)%P][co%Q],
                         name='output_vec', tag='im2col_conv_output')

    return output


def _schedule_im2col_conv2d(wkl, sch, s, data, data_pad, data_col, data_vec,
                            kernel, kernel_vec,
                            conv_out, output, last):
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)

    H, W = wkl.height, wkl.width
    CI = wkl.in_filter
    CO = wkl.out_filter
    HK, WK = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    HCAT, WCAT = HK-1, WK-1
    DOPAD = (HPAD != 0 and WPAD != 0)

    P = sch.vp
    Q = sch.vq
    UNROLL = sch.unroll

    A, B, C = data, kernel, last
    A0, A1, A2 = data_pad, data_col, data_vec
    B0 = kernel_vec
    C0, C1 = conv_out, output

    CC = s.cache_write(C0, "global")
    AA = s.cache_read(A2, "global", [CC])
    BB = s.cache_read(B0, "global", [CC])


    ##### Schedule CC
    _, co, im, vim, vco = s[C0].op.axis
    s[C0].unroll(vim)
    s[C0].vectorize(vco)

    s[CC].compute_at(s[C0], im)
    _, co, im, vim, vco = s[CC].op.axis
    ci, hk, wk = s[CC].op.reduce_axis
    s[CC].reorder(ci, hk, wk, vim, vco)
    s[CC].unroll(vim)
    s[CC].vectorize(vco)
    # s[CC].unroll(ccr)

    ### Schedule C
    _, co, h, w = s[C].op.axis
    im = s[C].fuse(h, w)
    im, vim = s[C].split(im, P)
    co, vco = s[C].split(co, Q)
    s[C].reorder(co, im, vim, vco)

    if sch.bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[C].split(co, sch.bc)
        oaxis = oco
        paxis = ico

    s[C].parallel(paxis)
    s[C].pragma(oaxis, "parallel_launch_point")
    s[C].pragma(paxis, "parallel_stride_pattern")
    s[C].pragma(oaxis, "parallel_barrier_when_finish")
    if C1 != C:
        s[C1].compute_inline()

    s[C0].compute_at(s[C], paxis)

    ##### Schedule A
    if DOPAD:
        s[A0].compute_inline()
    s[A1].compute_inline()
    s[AA].compute_at(s[CC], wk)
    s[AA].unroll(AA.op.axis[4])

    _, im, _, _, _, _ = s[A2].op.axis
    if sch.ba == 1:
        oaxis = im
        paxis = im
    else:
        oim, iim = s[A2].split(im, sch.ba)
        oaxis = oim
        paxis = iim

    s[A2].parallel(paxis)
    s[A2].pragma(oaxis, "parallel_launch_point")
    s[A2].pragma(paxis, "parallel_stride_pattern")
    s[A2].pragma(oaxis, "parallel_barrier_when_finish")


    ##### Schedule B
    s[BB].compute_at(s[CC], wk)
    s[BB].vectorize(BB.op.axis[4])

    co, _, _, _, _ = s[B0].op.axis
    if sch.bc == 1:
        oaxis = co
        paxis = co
    else:
        oco, ico = s[B0].split(co, sch.bc)
        oaxis = oco
        paxis = ico

    s[B0].parallel(paxis)
    s[B0].pragma(oaxis, "parallel_launch_point")
    s[B0].pragma(paxis, "parallel_stride_pattern")
    s[B0].pragma(oaxis, "parallel_barrier_when_finish")

    return s


def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    def check_device():
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
        W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

        out_dtype = 'float32'
        wkl, sch = _spatial_get_sch(A, W, stride, padding, out_dtype)

        a_shape = get_const_tuple(A.shape)
        w_shape = get_const_tuple(W.shape)

        dtype = A.dtype

        @memoize("topi.tests.test_topi_conv2d.verify_con2d_nchw")
        def get_ref_data():
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            w_np = np.random.uniform(size=w_shape).astype(dtype)
            b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
            c_np = np.maximum(b_np, 0)
            return a_np, w_np, b_np, c_np

        a_np, w_np, b_np, c_np = get_ref_data()
        # device = 'llvm'
        device = 'llvm -mcpu=skylake-avx512'
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)

        print('--- schedule data packing ---')
        A_vec, s = _spatial_pack_data_only(wkl, sch, A)
        print(A_vec.shape)
        a_vec_shape = get_const_tuple(A_vec.shape)
        a_vec = tvm.nd.array(np.zeros(a_vec_shape, dtype=dtype), ctx)
        print(tvm.lower(s, [A, A_vec], simple_mode=True))
        func = tvm.build(s, [A, A_vec], device)
        time_f = func.time_evaluator(func.entry_name, ctx, number=100)
        cost = time_f(a, a_vec).mean
        print('data -> data_vec: %g secs/op' % cost)

        print(A.shape)
        print(A_vec.shape)
        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=(device != "cuda")):
            # print('--- schedule data packing ---')
            # A_vec, s = _spatial_pack_data_only(wkl, sch, A)
            # print(A_vec.shape)
            # a_vec_shape = get_const_tuple(A_vec.shape)
            # a_vec = tvm.nd.array(np.zeros(a_vec_shape, dtype=dtype), ctx)
            # print(tvm.lower(s, [A, A_vec], simple_mode=True))
            # func = tvm.build(s, [A, A_vec], device)
            # time_f = func.time_evaluator(func.entry_name, ctx, number=100)
            # cost = time_f(a, a_vec).mean
            # print('data -> data_vec: %g secs/op' % cost)

            print('--- schedule kernel packing ---')
            W_vec, s = _spatial_pack_kernel_only(wkl, sch, W)
            print(W_vec.shape)
            w_vec_shape = get_const_tuple(W_vec.shape)
            w_vec = tvm.nd.array(np.zeros(w_vec_shape, dtype=dtype), ctx)
            print(tvm.lower(s, [W, W_vec], simple_mode=True))
            func = tvm.build(s, [W, W_vec], device)
            time_f = func.time_evaluator(func.entry_name, ctx, number=100)
            cost = time_f(w, w_vec).mean
            print('kernel -> kernel_vec: %g secs/op' % cost)

            print('--- schedule conv & unpack ---')
            A_vec = tvm.placeholder(a_vec_shape, name='A_vec')
            W_vec = tvm.placeholder(w_vec_shape, name='W_vec')
            B, s = _spatial_conv_only(wkl, sch, A_vec, W_vec, out_dtype=dtype)
            b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
            print(tvm.lower(s, [A_vec, W_vec, B], simple_mode=True))
            func = tvm.build(s, [A_vec, W_vec, B], target=device)
            func.save('conv_unpack.asm')
            time_f = func.time_evaluator(func.entry_name, ctx, number=2000)
            cost = time_f(a_vec, w_vec, b).mean
            print('conv & unpack: %g secs/op' % cost)

            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            print(b_np.shape)

    check_device()


def verify_conv2d_nchw_gemm(batch, in_channel, in_size, num_filter, kernel_size, stride, padding):
    in_height = in_width = in_size

    def check_device():
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
        W = tvm.placeholder((num_filter, in_channel, kernel_size, kernel_size), name='W')

        out_dtype = 'float32'
        wkl = _get_workload(A, W, stride, padding, out_dtype)
        sch = Im2ColPack(7, 8, 1, 8, True)

        a_shape = get_const_tuple(A.shape)
        w_shape = get_const_tuple(W.shape)

        dtype = A.dtype

        @memoize("topi.tests.test_topi_conv2d.verify_con2d_nchw")
        def get_ref_data():
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            w_np = np.random.uniform(size=w_shape).astype(dtype)
            b_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
            c_np = np.maximum(b_np, 0)
            return a_np, w_np, b_np, c_np

        a_np, w_np, b_np, c_np = get_ref_data()
        # device = 'llvm'
        device = 'llvm -mcpu=skylake-avx512'
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)

        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=(device != "cuda")):

            B = _im2col_pack(wkl, sch, A, W, stride, padding, out_dtype)
            s = tvm.create_schedule(B.op)
            traverse(s, B.op)

            op = B.op
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data_col = data_vec.op.input_tensors[0]
            data = data_col.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]
            _schedule_im2col_conv2d(wkl, sch, s, data, data_pad, data_col, data_vec,
                                    kernel, kernel_vec,
                                    conv_out, output, B)

            print(tvm.lower(s, [A, W, B], simple_mode=True))

            b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
            func = tvm.build(s, [A, W, B], device)
            func.save('conv.asm')
            time_f = func.time_evaluator(func.entry_name, ctx, number=2000)
            cost = time_f(a, w, b).mean
            print('conv: %g secs/op' % cost)

            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            print(b_np.shape)

    check_device()


def test_conv2d_nchw():
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1)
    # verify_conv2d_nchw_gemm(1, 64, 56, 64, 3, 1, 1)
    # ResNet18 worklaods
    """
    verify_conv2d_nchw(1, 3, 224, 64, 7, 2, 3)
    verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1)
    verify_conv2d_nchw(1, 64, 56, 64, 1, 1, 0)
    verify_conv2d_nchw(1, 64, 56, 128, 3, 2, 1)
    verify_conv2d_nchw(1, 64, 56, 128, 1, 2, 0)
    verify_conv2d_nchw(1, 128, 28, 128, 3, 1, 1)
    verify_conv2d_nchw(1, 128, 28, 256, 3, 2, 1)
    verify_conv2d_nchw(1, 128, 28, 256, 1, 2, 0)
    verify_conv2d_nchw(1, 256, 14, 256, 3, 1, 1)
    verify_conv2d_nchw(1, 256, 14, 512, 3, 2, 1)
    verify_conv2d_nchw(1, 256, 14, 512, 1, 2, 0)
    verify_conv2d_nchw(1, 512, 7, 512, 3, 1, 1)
    # Vgg16 workloads
    verify_conv2d_nchw(1, 128, 122, 128, 3, 1, 1)
    # Super resolution workloads
    verify_conv2d_nchw(1, 1, 224, 64, 5, 1, 2)
    verify_conv2d_nchw(1, 64, 224, 64, 3, 1, 1)
    verify_conv2d_nchw(1, 64, 224, 32, 3, 1, 1)
    verify_conv2d_nchw(1, 32, 224, 9, 3, 1, 1)
    """

if __name__ == "__main__":
    test_conv2d_nchw()
