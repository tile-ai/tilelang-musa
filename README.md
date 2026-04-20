# Tilelang_MUSA

Tile Language (**tile-lang**) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

Tilelang MUSA is a deeply adapted version of Tile Language for the MUSA platform and Moore Threads GPUs. It modifies the compilation pipeline (including Passes and Codegen) to target the MUSA toolchain and generate MUSA C that invokes `tl`-namespace templates under `src/tl_templates/musa`.

Tilelang MUSA supports almost all official Tilelang syntax, except for a small set of NVIDIA-specific features. Operators written with the Tilelang DSL can run on the MUSA platform through Tilelang MUSA with minimal migration effort.

In most cases, you only need to:
1. Set the decorator to `@tilelang.jit(target="musa")`, or use `@tilelang.jit`.
2. Use `torch_musa`, and create tensors with `device="musa"`.

<img src=./images/MatmulExample.png />

## Latest News

### 0.1.8+musa1

🚀 tilelang_musa0.1.8+musa1 has been released. This version is based on tilelang 0.1.8 and is deeply adapted for the musa platform. You can use the latest tilelang syntax to write kernels that can run on the musa platform. Our developers have already written some high-performance operators, including [GEMM](./testing/musa/mp31/basic/test_gemm.py), [FA](./testing/musa/mp31/flash_attention/), [DSA](./testing/musa/mp31/dsa/), etc.

## Tested Devices

Tilelang MUSA already supports S5000, S4000, and M1000.

## Build from Source

We recommend using a virtual environment.

```
conda create -n tilelang-dev python=3.10
conda activate tilelang-dev
```

Install Tilelang_MUSA

```
git clone --recursive https://github.com/MooreThreads/tilelang_musa.git
cd tilelang
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

## More Docs

For more examples, see the [MUSA MP31 Test Cases](./testing/musa/mp31/README.md) and [Common MUSA Test Cases](./testing/musa/common/README.md) documents.

See the official [Tilelang documentation](https://tilelang.com/) and [Tilelang MUSA Programming Guide](./docs/tilelang_musa_programming_guide.md).

## Acknowledgments

We would like to express our gratitude to the [Tilelang](https://github.com/tile-ai/tilelang) community and [TVM](https://github.com/apache/tvm) community for their invaluable contributions.
