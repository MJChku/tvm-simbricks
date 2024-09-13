# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _basic-mat-mult:

Simple Matrix Multiply
======================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

In this tutorial, we will build on top of the :ref:`vta-get-started` tutorial
and introduce additional concepts required to implement matrix multiplication
on VTA with the TVM workflow.
"""

######################################################################
# RPC Setup
# ---------
# We start by programming the Pynq's FPGA and building its RPC runtime
# as we did in the VTA introductory tutorial.

from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator
from vta.testing.simulator import _load_sw

from vta.libinfo import find_libvta
import time
import time
# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()
print(env.target)
# We read the Pynq RPC host IP address and port number from the OS environment
if False:
    host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
    port = int(os.environ.get("VTA_RPC_PORT", "9091"))

# _load_sw()
# lib_hw = find_libvta("libvta_hw", optional=True)
# assert lib_hw  # make sure to make in ${VTA_HW_PATH}/hardware/chisel
# f = tvm.get_global_func("vta.tsim.init")
# # m = tvm.runtime.load_module(lib_hw[0], "vta-tsim")

# exit()

    tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    remote = tracker.request('tsim', priority=1, session_timeout=30)
print("Connection Success")
#exit()

remote = rpc.LocalSession()

# Output channel factor m - total 16x16=256 output channels
m = 16
# Input channel factor n - total 16x16=256 input channels
n = 16
# Batch factor o (we use single batch inference)
o = 1
# A placeholder tensor in tiled data format
A = te.placeholder((o, n, env.BATCH, env.BLOCK_IN), name="A", dtype=env.inp_dtype)
# B placeholder tensor in tiled data format
B = te.placeholder((m, n, env.BLOCK_OUT, env.BLOCK_IN), name="B", dtype=env.wgt_dtype)
# A copy buffer
A_buf = te.compute((o, n, env.BATCH, env.BLOCK_IN), lambda *i: A(*i), "A_buf")
# B copy buffer
B_buf = te.compute((m, n, env.BLOCK_OUT, env.BLOCK_IN), lambda *i: B(*i), "B_buf")

######################################################################
# Matrix Multiplication
# ~~~~~~~~~~~~~~~~~~~~~
# Now we're ready to describe the matrix multiplication result tensor :code:`C`,
# with another compute operation.
# The compute function takes the shape of the tensor, as well as a lambda
# function that describes the computation rule for each position of the tensor.
#
# In order to implement matrix multiplication, the lambda function needs to
# include a reduction formula over the input channel dimension axes.
# To create a reduction formula, we can declare a reduction axis using
# :code:`te.reduce_axis`, which takes in the range of reductions.
# :code:`te.sum` takes in the expression to be reduced as well as
# the reduction axes to compute the sum of value over all k in the declared
# ranges.
#
# Note that the reduction needs to be performed over 32-bit :code:`env.acc_dtype`
# accumulator data types.
#
# No computation happens during this phase, as we are only declaring how
# the computation should be done.

# Outer input feature reduction axis
start_time = time.time()
ko = te.reduce_axis((0, n), name="ko")
# Inner input feature reduction axis
ki = te.reduce_axis((0, env.BLOCK_IN), name="ki")
# Describe the in-VTA matrix multiplication
C_buf = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT),
    lambda bo, co, bi, ci: te.sum(
        A_buf[bo, ko, bi, ki].astype(env.acc_dtype) * B_buf[co, ko, ci, ki].astype(env.acc_dtype),
        axis=[ko, ki],
    ),
    name="C_buf",
)

# Cast to output type, and send to main memory
C = te.compute(
    (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: C_buf(*i).astype(env.inp_dtype), name="C"
)

# Let's take a look at the generated schedule
s = te.create_schedule(C.op)
# print(tvm.lower(s, [A, B, C], simple_mode=True))

# exit()

# Set the intermediate tensor's scope to VTA's on-chip buffers
s[A_buf].set_scope(env.inp_scope)
s[B_buf].set_scope(env.wgt_scope)
s[C_buf].set_scope(env.acc_scope)

# Move buffer copy into matrix multiply loop
s[A_buf].compute_at(s[C_buf], ko)
s[B_buf].compute_at(s[C_buf], ko)

# Tag the buffer copies with the DMA pragma to insert a DMA transfer
s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
s[C].pragma(s[C].op.axis[0], env.dma_copy)

# Let's take a look at the transformed schedule
# print(tvm.lower(s, [A, B, C], simple_mode=True))

s[C_buf].reorder(
    ko, s[C_buf].op.axis[0], s[C_buf].op.axis[1], s[C_buf].op.axis[2], s[C_buf].op.axis[3], ki
)
s[C_buf].tensorize(s[C_buf].op.axis[2], env.gemm)

# Build GEMM VTA kernel
my_gemm = vta.build(
    s, [A, B, C], tvm.target.Target("ext_dev", host=env.target_host), name="my_gemm"
)

# Write the compiled module into an object file.
temp = utils.tempdir()
my_gemm.save(temp.relpath("gemm.o"))

# print("starts to sleep")
# time.sleep(5)
# print("ends to sleep")
# Send the executable over RPC
# start_time = time.time()
remote.upload(temp.relpath("gemm.o"))
# Load the compiled module
f = remote.load_module("gemm.o")
# end_time = time.time()


######################################################################
# Running the Function
# --------------------
# The compiled TVM function uses a concise C API and can be invoked from
# code language.
#
# TVM provides an array API in python to aid quick testing and prototyping.
# The array API is based on `DLPack <https://github.com/dmlc/dlpack>`_ standard.
#
# - We first create a remote context (for remote execution on the Pynq).
# - Then :code:`tvm.nd.array` formats the data accordingly.
# - :code:`f()` runs the actual computation.
# - :code:`numpy()` copies the result array back in a format that can be
#   interpreted.
#

# Get the remote device context
ctx = remote.ext_dev(0)
print("ctx", ctx)
# Initialize the A and B arrays randomly in the int range of (-128, 128]
A_orig = np.random.randint(-128, 128, size=(o * env.BATCH, n * env.BLOCK_IN)).astype(A.dtype)
B_orig = np.random.randint(-128, 128, size=(m * env.BLOCK_OUT, n * env.BLOCK_IN)).astype(B.dtype)

# Apply packing to the A and B arrays from a 2D to a 4D packed layout
A_packed = A_orig.reshape(o, env.BATCH, n, env.BLOCK_IN).transpose((0, 2, 1, 3))
B_packed = B_orig.reshape(m, env.BLOCK_OUT, n, env.BLOCK_IN).transpose((0, 2, 1, 3))

# Format the input/output arrays with tvm.nd.array to the DLPack standard
A_nd = tvm.nd.array(A_packed, ctx)
B_nd = tvm.nd.array(B_packed, ctx)
C_nd = tvm.nd.array(np.zeros((o, m, env.BATCH, env.BLOCK_OUT)).astype(C.dtype), ctx)

# Clear stats
if env.TARGET in ["sim", "tsim"]:
    simulator.clear_stats()

print("Clearned already ")

print("Start to run the function ")
# Sleep for 1 second
# Invoke the module to perform the computation
# Measure start time

start_time = time.time()

# Invoke the module to perform the computation
#for i in np.arange(1, 100):
#    f(A_nd, B_nd, C_nd)

f(A_nd, B_nd, C_nd)
# Measure end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print elapsed time

print("Elapsed time:", elapsed_time)
C_ref = np.dot(A_orig.astype(env.acc_dtype), B_orig.T.astype(env.acc_dtype)).astype(C.dtype)
C_ref = C_ref.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))
np.testing.assert_equal(C_ref, C_nd.numpy())

# Print stats
time.sleep(1)

if env.TARGET in ["sim", "tsim"]:
    sim_stats = simulator.stats()
    print("Execution statistics:")
    for k, v in sim_stats.items():
        print("\t{:<16}: {:>16}".format(k, v))

print("Successful matrix multiply test!")

######################################################################
# Summary
# -------
# This tutorial showcases the TVM workflow to implement a simple matrix
# multiplication example on VTA.
# The general workflow includes:
#
# - Programming the FPGA with the VTA bitstream over RPC.
# - Describing matrix multiplication via a series of computations.
# - Describing how we want to perform the computation using schedule primitives.
# - Compiling the function to the VTA target.
# - Running the compiled module and verifying it against a numpy implementation.
#
