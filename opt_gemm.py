import tvm
import numpy
import time

# TVM_NUM_THREADS=4 python2 opt_gemm.py

# The size of the square matrix
N = 1024
# The default tensor type in tvm
dtype = "float32"
target = "llvm -mcpu=skylake-avx512"
# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
b = tvm.nd.array(numpy.random.rand(N, N).astype(dtype), tvm.cpu(0))
# The expected answer
answer = numpy.dot(a.asnumpy(), b.asnumpy())

# Algorithm
k = tvm.reduce_axis((0, N), 'k')
A = tvm.placeholder((N, N), name = 'A')
B = tvm.placeholder((N, N), name = 'B')

bn = 8
packedB = tvm.compute((N / bn, N, bn), lambda x, y, z: B[y, x * bn + z], name = 'packedB')
C = tvm.compute(A.shape,
                lambda x, y: tvm.sum(A[x, k] * packedB[y / bn, k, y % bn], axis = k),
                name = 'C')

# Same schedule
s = tvm.create_schedule(C.op)
x, y, z = s[packedB].op.axis
# s[packedB].reorder(y, x, z)
s[packedB].vectorize(z)

# CC = s.cache_write(C, 'local')
AA = s.cache_read(A, 'local', [C])
# BB = s.cache_read(packedB, 'local', [C])

yo, xo, yi, xi = s[C].tile(C.op.axis[1], C.op.axis[0], bn, bn)
ko, ki = s[C].split(k, factor=4)

s[C].reorder(yo, xo, ko, ki, yi, xi)
s[C].vectorize(yi)
s[C].unroll(xi)
s[C].unroll(ki)

# s[CC].compute_at(s[C], xo)
# yi, xi = s[CC].op.axis
# k, = s[CC].op.reduce_axis
# ko, ki = s[CC].split(k, factor=4)
#
# s[CC].reorder(ko, ki, yi, xi)
# s[CC].vectorize(yi)
# s[CC].unroll(ki)
# s[CC].unroll(xi)

s[AA].compute_at(s[C], ko)
ao, ai = s[AA].op.axis
s[AA].vectorize(ai)
s[AA].unroll(ao)

# s[BB].compute_at(s[C], ko)
# print(s[BB].op.axis)
# _, bo, bi = s[BB].op.axis
# s[BB].vectorize(bi)
# s[BB].unroll(bo)

s[C].parallel(yo)

print(tvm.lower(s, [A, B, C], simple_mode=True))
func = tvm.build(s, [A, B, C], target=target, name = 'mmult')
assert func
# We can accelerate it almost 3x compared with the previous schedule.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 100)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Opt3: %f' % evaluator(a, b, c).mean)

_a = a.asnumpy()
_b = b.asnumpy()
now = time.time()
answer = numpy.dot(_a, _b)
print("Numpy: %f" % (time.time() - now))

