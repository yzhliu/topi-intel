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
import math

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


def _spatial_conv_all(wkl, sch, data, kernel, out_dtype):
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

    dshape = (1, CI, H, W)
    dpshape = (1, CI, TH, TW)
    dvshape = (1, TH // (VH * HSTR), TW // (VW * WSTR), CI, VH * HSTR + HCAT, VW * WSTR + WCAT)

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw: \
        data_pad[n][ci][h * VH * HSTR + vh][w * VW * WSTR + vw], name='data_vec')

    kshape = (CO, CI, KH, KW)
    kvshape = (CO // VC, CI, KH, KW, VC)

    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc: \
        kernel[co * VC + vc][ci][dh][dw], name='kernel_vec')

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

    s = tvm.create_schedule(conv.op)
    traverse(s, conv.op)

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

    # schedule for kernel_vec
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

    # schedule for conv & unpack
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


def verify_conv2d_nchw(sch, batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    def check_device():
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
        W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

        out_dtype = 'float32'

        wkl = _get_workload(A, W, stride, padding, out_dtype)

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
        device = 'llvm -mcpu=skylake-avx512'
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)

        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=(device != "cuda")):
            A_vec, s = _spatial_pack_data_only(wkl, sch, A)
            a_vec_shape = get_const_tuple(A_vec.shape)
            a_vec = tvm.nd.array(np.zeros(a_vec_shape, dtype=dtype), ctx)
            func = tvm.build(s, [A, A_vec], device)
            time_f = func.time_evaluator(func.entry_name, ctx, number=20)
            cost_data = time_f(a, a_vec).mean

            W_vec, s = _spatial_pack_kernel_only(wkl, sch, W)
            w_vec_shape = get_const_tuple(W_vec.shape)
            w_vec = tvm.nd.array(np.zeros(w_vec_shape, dtype=dtype), ctx)
            func = tvm.build(s, [W, W_vec], device)
            time_f = func.time_evaluator(func.entry_name, ctx, number=20)
            cost_kernel = time_f(w, w_vec).mean

            A_vec = tvm.placeholder(a_vec_shape, name='A_vec')
            W_vec = tvm.placeholder(w_vec_shape, name='W_vec')
            B, s = _spatial_conv_only(wkl, sch, A_vec, W_vec, out_dtype=dtype)
            b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
            func = tvm.build(s, [A_vec, W_vec, B], target=device)
            time_f = func.time_evaluator(func.entry_name, ctx, number=20)
            cost_conv = time_f(a_vec, w_vec, b).mean

            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            return (cost_data, cost_kernel, cost_conv)

    return check_device()


def verify_conv2d_nchw_all(sch, batch, in_channel, in_size, num_filter, kernel, stride, padding):
    in_height = in_width = in_size

    def check_device():
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
        W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

        out_dtype = 'float32'

        wkl = _get_workload(A, W, stride, padding, out_dtype)

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
        device = 'llvm -mcpu=skylake-avx512'
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)

        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=(device != "cuda")):
            B, s = _spatial_conv_all(wkl, sch, A, W, out_dtype=dtype)
            b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
            # print(tvm.lower(s, [A_vec, W_vec, B], simple_mode=True))
            func = tvm.build(s, [A, W, B], target=device)
            time_f = func.time_evaluator(func.entry_name, ctx, number=2000)
            cost = time_f(a, w, b).mean

            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            return cost

    return check_device()


def factors(n):
    # 1 and n are automatically factors of n
    fact=[1,n]
    # starting at 2 as we have already dealt with 1
    check=2
    # calculate the square root of n and use this as the
    # limit when checking if a number is divisible as
    # factors above sqrt(n) will already be calculated as
    # the inverse of a lower factor IE. finding factors of
    # 100 only need go up to 10 (sqrt(100)=10) as factors
    # such as 25 can be found when 5 is found to be a
    # factor 100/5=25
    rootn = math.sqrt(n)
    while check<rootn:
        if n % check == 0:
            fact.append(check)
            fact.append(n//check)
        check+=1
    # this line checks the sqrt of the number to see if
    # it is a factor putting it here prevents it appearing
    # twice in the above while loop and putting it outside
    # the loop should save some time.
    if rootn==check:
        fact.append(check)
    # return an array of factors sorted into numerial order.
        fact.sort()
    return fact


def test_conv2d_nchw():
    cost_all = []
    cost_data = []
    cost_kernel = []
    cost_conv = []
    schedules = []

    # g1mb1_ic3ih224iw224_oc64oh112ow112_kh7kw7_sh2sw2_ph3pw3_n : 0.773982 ms
    in_channel, in_size, num_filter, kernel, stride, padding = 3, 224, 64, 7, 2, 3

    # g1mb1_ic64ih56iw56_oc64oh56ow56_kh3kw3_sh1sw1_ph1pw1_n : 0.72514 ms
    # in_channel, in_size, num_filter, kernel, stride, padding = 64, 56, 64, 3, 1, 1

    out_size = (in_size + 2 * padding - kernel) // stride + 1
    vhw_candidates = factors(out_size)
    vc_candidates = factors(num_filter)
    ba_candidates = factors(in_size)
    bc_candidates = factors(kernel)

    with open('report/conv_search.txt', 'a') as report:
        while True:
            vhw_idx = np.random.randint(0, len(vhw_candidates), size=(2,))
            vc_idx = np.random.randint(0, len(vc_candidates))
            ba_idx = np.random.randint(0, len(ba_candidates))
            bc_idx = np.random.randint(0, len(bc_candidates))

            sch = SpatialPack(vh=vhw_candidates[vhw_idx[0]], vw=vhw_candidates[vhw_idx[1]],
                              vc=vc_candidates[vc_idx], ba=ba_candidates[ba_idx], bc=bc_candidates[bc_idx],
                              unroll=bool(np.random.randint(0, 2)))
            try:
                print("Trying " + str(sch))
                cost = verify_conv2d_nchw_all(sch, 1, in_channel, in_size, num_filter, kernel, stride, padding)
                data_cost, kernel_cost, conv_cost = verify_conv2d_nchw(sch, 1, in_channel, in_size, num_filter, kernel, stride, padding)
                print("Successful try with %s, conv total time = %f" % (str(sch), cost))

                cost_all.append(cost)
                cost_data.append(data_cost)
                cost_kernel.append(kernel_cost)
                cost_conv.append(conv_cost)
                schedules.append(sch)

                if len(cost_conv) >= 20:
                    idx = np.argmin(cost_all)
                    report.write('%s\tall=%f\tconv=%f\tdata=%f\tkernel=%f\n' %
                                 (str(sch), cost_all[idx], cost_conv[idx], cost_data[idx], cost_kernel[idx]))
                    report.flush()
                    cost_all = []
                    cost_conv = []
                    cost_kernel = []
                    cost_data = []
            except Exception as e:
                print(e)


if __name__ == "__main__":
    test_conv2d_nchw()
