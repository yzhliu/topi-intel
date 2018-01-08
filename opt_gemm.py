import tvm
import numpy
import time

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
bn = 8
k = tvm.reduce_axis((0, N), 'k')
A = tvm.placeholder((N, N), name = 'A')
B = tvm.placeholder((N, N), name = 'B')
# We have to re-write the algorithm slightly.
packedB = tvm.compute((N / bn, N, bn), lambda x, y, z: B[y, x * bn + z], name = 'packedB')
C = tvm.compute(A.shape,
                lambda x, y: tvm.sum(A[x, k] * packedB[y / bn, k, y % bn], axis = k),
                name = 'C')

# Same schedule
s = tvm.create_schedule(C.op)
yo, xo, yi, xi = s[C].tile(C.op.axis[1], C.op.axis[0], bn, bn)
s[C].reorder(yo, xo, k, yi, xi)
s[C].vectorize(yi)

func = tvm.build(s, [A, B, C], target=target, name = 'mmult')
assert func
# We can accelerate it almost 3x compared with the previous schedule.
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
c = tvm.nd.array(numpy.zeros((N, N), dtype = dtype), tvm.cpu(0))
print('Opt3: %f' % evaluator(a, b, c).mean)

##################################################################################################
# Summary
# -------
# After applying three main tricks, we can achieve almost 90% performance of numpy.
# Further observation is required to catch up with the performance of numpy.
#

# TODO(Jian Weng): Catch up with the performance of numpy.
_a = a.asnumpy()
_b = b.asnumpy()
now = time.clock()
answer = numpy.dot(_a, _b)
print("Numpy: %f" % (time.clock() - now))

