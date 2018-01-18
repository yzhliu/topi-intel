import tvm
import numpy

N = 1024
bn = 8
dtype='float32'
times = 50
device = 'llvm -mcpu=skylake-avx512'

def run_together():
    A = tvm.placeholder((N, N), name='A')
    B = tvm.compute((N // bn, N, bn), lambda x, y, z: A[y, x * bn + z], name='packedB')
    k = tvm.reduce_axis((0, N), 'k')
    C = tvm.compute(A.shape,
                    lambda x, y: tvm.sum(A[x, k] * B[y / bn, k, y % bn], axis=k),
                    name='C')

    s = tvm.create_schedule(C.op)
    func = tvm.build(s, [A, C], device)

    print(tvm.lower(s, [A, C], simple_mode=True))

    a = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
    c = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))

    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=times)
    print('time: %f sec' % evaluator(a, c).mean)


def run_separate():
    A = tvm.placeholder((N, N), name = 'A')
    B = tvm.compute((N // bn, N, bn), lambda x, y, z: A[y, x * bn + z], name = 'packedB')
    s = tvm.create_schedule(B.op)
    func = tvm.build(s, [A, B], device)
    print(tvm.lower(s, [A, B], simple_mode=True))
    a = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
    b = tvm.nd.array(numpy.random.rand(N//bn, N, bn).astype(dtype), tvm.cpu(0))
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=times)
    print('time pack: %f sec' % evaluator(a, b).mean)

    k = tvm.reduce_axis((0, N), 'k')
    A = tvm.placeholder((N, N), name = 'A')
    B = tvm.placeholder((N // bn, N, bn), name = 'B')
    C = tvm.compute(A.shape,
                    lambda x, y: tvm.sum(A[x, k] * B[y / bn, k, y % bn], axis=k),
                    name='C')
    s = tvm.create_schedule(C.op)
    c = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
    func = tvm.build(s, [A, B, C], device)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number=times)
    print('time gemm: %f sec' % evaluator(a, b, c).mean)

print('Run together ...')
run_together()

print('Run separately ...')
run_separate()
