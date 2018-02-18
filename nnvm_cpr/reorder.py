import nnvm.compiler
import nnvm.symbol as sym

import tvm
from tvm.contrib import graph_runtime

import numpy as np

# (oc, ic, h, w)
oc, ic, h, w = 3, 4, 2, 2
shape = (oc, ic, h, w)
oc_bn, ic_bn = 3, 2
# (OC, IC, h, w, ic, oc)
out_shape = (oc//oc_bn, ic//ic_bn, h, w, ic_bn, oc_bn)

x = sym.Variable("x")
# y = sym.Variable("y")
# z = sym.elemwise_add(x, sym.sqrt(y))
# z = sym.reshape(x, shape=out_shape)
z = sym.reorder(x, oc_bn=3, ic_bn=2)
compute_graph = nnvm.graph.create(z)
print("-------compute graph-------")
print(compute_graph.ir())

deploy_graph, lib, params = nnvm.compiler.build(
    compute_graph, target="llvm", shape={"x": shape}, dtype="float32")

module = graph_runtime.create(deploy_graph, lib, tvm.cpu(0))
x_np = np.random.uniform(0, 255, size=shape).astype("float32")
print(x_np)
# y_np = np.array([[4, 4], [4, 4], [4, 4]]).astype("float32")
# set input to the graph module
module.set_input(x=x_np) #, y=y_np)
# run forward computation
module.run()
# get the first output
out = module.get_output(0, out=tvm.nd.empty(out_shape))
print(out.asnumpy())
