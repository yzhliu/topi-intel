import tvm
import numpy as np

# The size of the square matrix
N = 1024
nstep = 1<<16
# The default tensor type in tvm
dtype = "float32"
target = "llvm -mcpu=skylake-avx512"
# target = "llvm"
# Random generated tensor for testing
a = tvm.nd.array(np.random.rand(N,).astype(dtype), tvm.cpu(0))
c = tvm.nd.array(np.zeros((N,), dtype = dtype), tvm.cpu(0))

# The expected answer
answer = a.asnumpy() * nstep

x = tvm.placeholder((N,), name="x")
k = tvm.reduce_axis((0, nstep))
y = tvm.compute((N,), lambda i: tvm.sum(x[i], axis=k), name="y")

s = tvm.create_schedule(y.op)

i, = s[y].op.axis
io, ii = s[y].split(i, factor=32)
s[y].vectorize(ii)

print(tvm.lower(s, [x, y], simple_mode=True))

func = tvm.build(s, [x, y], target=target, name = 'adddd')
assert func
func.save('trivial.asm')

evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 50)
print('Time: %f' % evaluator(a, c).mean)

np.testing.assert_allclose(c.asnumpy(), answer, rtol=1e1)
