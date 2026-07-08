#!/usr/bin/env sh
# Pin the CUDA toolkit lookup to the activated environment's own CUDA toolchain.
#
# torch's JIT extension build (cpp_extension) follows CUDA_HOME to locate nvcc
# and the CUDA headers/libs. Pointing CUDA_HOME at the environment prefix makes
# JIT-compiled kernels use the same CUDA version the environment ships, keeping
# the compiler in sync with the runtime libraries already linked into PyTorch --
# a compiler/runtime version skew otherwise causes kernel linking to fail.
#
# Scoped to the active environment: the override applies only while the env is
# activated and does not alter the shell's global CUDA_HOME.
export CUDA_HOME="${CONDA_PREFIX}"
