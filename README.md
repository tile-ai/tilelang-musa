# TileLang MUSA

Tile Language (**tile-lang**) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

Tilelang MUSA is a deeply adapted version of Tile Language for the MUSA platform and Moore Threads GPUs. It modifies the compilation pipeline (including Passes and Codegen) to target the MUSA toolchain and generate MUSA C that invokes `tl`-namespace templates under `src/tl_templates/musa`.

Tilelang MUSA supports almost all official Tilelang syntax, except for a small set of NVIDIA-specific features. Operators written with the Tilelang DSL can run on the MUSA platform through Tilelang MUSA with minimal migration effort.

In most cases, you only need to:
1. Set the decorator to `@tilelang.jit(target="musa")`, or use `@tilelang.jit`.
2. Use `torch_musa`, and create tensors with `device="musa"`.

<img src=./images/MatmulExample.png />

## Version and Installation

| Component | Version | Purpose |
| --- | --- | --- |
| Public source baseline | [`v0.1.12+musa.1`](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1) | Source code and API reference in this repository |
| Recommended binary package | `tilelang_musa==0.1.12+musa.5` | Install from the Moore Threads Python package index |

The public source baseline and the recommended binary package have different revision numbers. Building this repository produces the public `.1` source version; it does not reproduce the `.5` binary package.

### Install the Binary Package

Use a Linux x86_64 environment with Python 3.10 or later. Prepare a compatible MUSA SDK, GPU driver, PyTorch and `torch_musa` environment first, then install:

```bash
python -m pip install 'tilelang_musa==0.1.12+musa.5' -i https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

The package is imported as `tilelang`. Use a separate virtual environment from upstream PyPI `tilelang`, which shares that import name.

Check the installed package version and MUSA device visibility:

```bash
python -c "from importlib.metadata import version; import torch, torch_musa, tilelang; print('tilelang_musa:', version('tilelang_musa')); print('tilelang:', tilelang.__version__); print('MUSA available:', torch.musa.is_available())"
```

This check reports installation and device visibility; run an operator correctness test before using your workload.

## Latest News

### 0.1.12+musa.5 Binary Package

The recommended binary installation is now `tilelang_musa==0.1.12+musa.5`, available from the Moore Threads package index above. The public source baseline remains `v0.1.12+musa.1`.

See [TileOPs v1.0.0](https://github.com/MooreThreads/TileOPs/releases/tag/v1.0.0) for MUSA operator implementations and development examples.

### 0.1.10+musa.1

🚀 tilelang_musa0.1.10+musa.1 has been released. This version is based on tilelang 0.1.10 and is deeply adapted for the musa platform. You can use the latest tilelang syntax to write kernels that can run on the musa platform.

### 0.1.9+musa.1

🚀 tilelang_musa0.1.9+musa.1 has been released. This version is based on tilelang 0.1.9 and is deeply adapted for the musa platform. You can use the latest tilelang syntax to write kernels that can run on the musa platform.

### 0.1.8+musa.3

Add support for DeepSeek [TileKernels](./tilekernels/) on the MUSA platform. The upstream repository is https://github.com/deepseek-ai/TileKernels.

### 0.1.8+musa.1

🚀 tilelang_musa0.1.8+musa.1 has been released. This version is based on tilelang 0.1.8 and is deeply adapted for the musa platform. You can use the latest tilelang syntax to write kernels that can run on the musa platform. Our developers have already written some high-performance operators, including [GEMM](./testing/musa/mp31/basic/test_gemm.py), [FA](./testing/musa/mp31/flash_attention/), [DSA](./testing/musa/mp31/dsa/), etc.

## Tested Devices

Tilelang MUSA already supports S5000, S4000, and M1000.

## Build from Public Source

To build the published `v0.1.12+musa.1` source version, use the exact tag below. To use the recommended `.5` binary, follow [Install the Binary Package](#install-the-binary-package).

We recommend using a virtual environment.

```
conda create -n tilelang-dev python=3.10
conda activate tilelang-dev
```

Install Tilelang_MUSA

```
git clone --branch v0.1.12+musa.1 --recursive https://github.com/tile-ai/tilelang-musa.git
cd tilelang-musa
pip install -r ./requirements-dev.txt
pip install -e . -v --no-build-isolation
```

## Quick Start

In this section, you'll learn how to write and execute a straightforward GEMM (matrix multiplication) kernel using tile-lang, followed by techniques for layout optimizations, pipelining, and L2-cache–friendly swizzling.

### GEMM Example with Annotations (Layout, L2 Cache Swizzling, and Pipelining, etc.)

Below is an example that demonstrates more advanced features: layout annotation, parallelized copy, and swizzle for improved L2 cache locality. This snippet shows how to adapt your kernel to maximize performance on complex hardware.

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="musa")
def matmul(A, B, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # Enable rasterization for better L2 cache locality (Optional)
        # T.use_swizzle(panel_size=4, order='col')

        # Clear local accumulation
        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            # Copy tile of A
            # This is a sugar syntax for parallelized copy
            T.copy(A[by * block_M, ko * block_K], A_shared)

            # Copy tile of B
            T.copy(B[ko * block_K, bx * block_N], B_shared)

            # Perform a tile-level GEMM on the shared buffers
            # Dispatch to target-specific GEMM backend implementation
            T.gemm(A_shared, B_shared, C_local)

        # relu
        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        # Copy result back to global memory
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


M = 1024  # M = T.dynamic("m") if you want to use dynamic shape
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 64

# 1. Define the kernel (matmul) and compile/lower it into an executable module
kernel = matmul.compile(
    M=M,
    N=N,
    K=K,
    block_M=block_M,
    block_N=block_N,
    block_K=block_K
)

# 2. Test the kernel in Python with PyTorch data
import torch
import torch_musa

# Create random input tensors on the GPU
a = torch.randn(M, K, device="musa", dtype=torch.float16)
b = torch.randn(K, N, device="musa", dtype=torch.float16)

# Run the kernel through the Profiler
c = kernel(a, b)

print(c)

# Reference multiplication using PyTorch
ref_c = torch.relu(a @ b)

# Validate correctness
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

# 3. Retrieve and inspect the generated MUSA source (optional)
# musa_source = kernel.get_kernel_source()
# print("Generated MUSA kernel:\n", musa_source)

# 4.Profile latency with kernel
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

latency = profiler.do_bench()

print(f"Latency: {latency} ms")
```

## Operator Development References / 算子开发参考

### TileOPs: Operator Implementations and Examples

[MooreThreads/TileOPs](https://github.com/MooreThreads/TileOPs/tree/v1.0.0) provides MUSA operator implementations built on TileLang, with operator interfaces, kernel implementations, tests and benchmarks. It is a useful reference for developing and integrating operators on MP31 / MTT S5000.

- [TileOPs v1.0.0 release notes](https://github.com/MooreThreads/TileOPs/releases/tag/v1.0.0): environment setup, operator coverage, API usage and release scope.
- [Operator and kernel source](https://github.com/MooreThreads/TileOPs/tree/v1.0.0/src/tileops): examples across GEMM, attention, normalization, reductions, MoE and other operator families.
- [Tests](https://github.com/MooreThreads/TileOPs/tree/v1.0.0/tests) and [benchmarks](https://github.com/MooreThreads/TileOPs/tree/v1.0.0/benchmarks): reference inputs, correctness checks and measurement examples.

TileOPs v1.0.0 documents `0.1.12+musa.4` as its compiler runtime. This README recommends `.5` for TileLang-MUSA installation; check your selected TileOPs operators in that environment. Consult the release notes for implementation status and validation scope.

### Programming, Validation and Optimization

| Development topic | Public reference |
| --- | --- |
| Kernel syntax and the basic development flow | [Quick Start](#quick-start) and [upstream TileLang documentation](https://tilelang.com/) |
| MUSA copy, synchronization and extension semantics | [TileLang MUSA Programming Guide](./docs/tilelang_musa_programming_guide.md) |
| Elementwise, reductions and basic kernel behavior | [Common MUSA test cases](./testing/musa/common/README.md) |
| GEMM, attention, pipeline and layout examples | [MP31 test cases and operator examples](./testing/musa/mp31/README.md) |
| Operator interfaces, integration and measurement examples | [TileOPs v1.0.0](https://github.com/MooreThreads/TileOPs/releases/tag/v1.0.0) |

Start with a small input and compare against a PyTorch reference. Check shapes, dtypes, layouts, boundary conditions and synchronization before tuning tile sizes, pipelines or memory layouts. Measure performance after correctness passes, and record the hardware and software versions with the results.

The local guide and test references describe the public `.1` source baseline. Account for revision differences when using the `.5` binary or examples from newer upstream documentation.

## More Docs

For more examples, see the [MUSA MP31 Test Cases](./testing/musa/mp31/README.md) and [Common MUSA Test Cases](./testing/musa/common/README.md) documents.

See the official [Tilelang documentation](https://tilelang.com/) and [Tilelang MUSA Programming Guide](./docs/tilelang_musa_programming_guide.md).

## Acknowledgments

We would like to express our gratitude to the [Tilelang](https://github.com/tile-ai/tilelang) community and [TVM](https://github.com/apache/tvm) community for their invaluable contributions.
