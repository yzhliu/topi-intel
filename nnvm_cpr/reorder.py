import nnvm.compiler
import nnvm.symbol as sym

import tvm
from tvm.contrib import graph_runtime

import numpy as np

x = sym.Variable("x")
# y = sym.Variable("y")
# z = sym.elemwise_add(x, sym.sqrt(y))
z = sym.transpose(x)
compute_graph = nnvm.graph.create(z)
print("-------compute graph-------")
print(compute_graph.ir())

shape = (3, 2)
out_shape = (2, 3)
deploy_graph, lib, params = nnvm.compiler.build(
    compute_graph, target="llvm", shape={"x": shape}, dtype="float32")

module = graph_runtime.create(deploy_graph, lib, tvm.cpu(0))
x_np = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
print(x_np)
# y_np = np.array([[4, 4], [4, 4], [4, 4]]).astype("float32")
# set input to the graph module
module.set_input(x=x_np) #, y=y_np)
# run forward computation
module.run()
# get the first output
out = module.get_output(0, out=tvm.nd.empty(out_shape))
print(out.asnumpy())
