---
title: 安装
description: TileLang-MUSA Python 包安装与软件源配置
tags: [MUSA, TileLang]
---

## 安装

### 配置 pip 软件源

请根据使用场景，选择以下任一种方式配置 pip 软件源：

1. 为当前 shell 会话或 CI 任务配置软件源：

   ```bash
   export PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
   ```

2. 将软件源写入 pip 配置文件，并查看当前配置：

   ```bash
   python -m pip config set global.index-url \
     https://dl.mthreads.com/repo/api/pypi/pypi/simple
   python -m pip config list
   ```

3. 不修改 pip 配置，在安装时指定软件源：

   ```bash
   python -m pip install 'tilelang_musa==0.1.12+musa.5' \
     -i https://dl.mthreads.com/repo/api/pypi/pypi/simple
   python -m pip install 'apache-tvm-ffi==0.1.11.post1' \
     --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
   ```

:::note
请勿在同一次安装中同时使用摩尔线程 Python 包源和公开 PyPI 镜像源。安装普通第三方包时，请切换软件源后单独安装。
:::

### 安装 Python 包

使用环境变量或 pip 配置文件完成配置后，可通过 pip 直接安装 TileLang-MUSA：

```bash
python -m pip install 'tilelang_musa==0.1.12+musa.5'
python -m pip install 'apache-tvm-ffi==0.1.11.post1'
```

安装完成后仍通过 `import tilelang` 使用。

### 从源码构建

TileLang-MUSA 源码仓库位于
[https://github.com/tile-ai/tilelang-musa](https://github.com/tile-ai/tilelang-musa)。
本版本的源码 tag 为 `v0.1.12+musa.5`，MUSA 部分覆盖 PH1/MP31 与 QY2/MP22。
先准备[版本与环境](./01_version_environment.md)中的 Linux、MUSA SDK、驱动、
PyTorch、`torch_musa` 和 `apache-tvm-ffi==0.1.11.post1`。源码构建还需要
C/C++ 编译工具、Python 开发头文件、CMake 3.26.1 或更新版本，以及 Ninja 或 Make。

```bash
git clone --branch 'v0.1.12+musa.5' --recursive https://github.com/tile-ai/tilelang-musa.git
cd tilelang-musa
git submodule status --recursive

# 在已准备好 MUSA SDK 和 Python 依赖的环境中构建
USE_MUSA=ON USE_CUDA=OFF \
  CMAKE_ARGS='-DUSE_MUSA=ON -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release' \
  python -m pip install -v .
```

TVM 固定到公开提交
[`643f4919007cf2201a9d2fac4f51e6373d5b537f`](https://github.com/MooreThreads/tvm_musa/commit/643f4919007cf2201a9d2fac4f51e6373d5b537f)。
不要用 TVM 的最新分支替换该提交。`--recursive` 还会检出 MUTLASS 和 TVM 的嵌套
依赖；GitHub 自动生成的源码 zip/tar 不包含这些 submodule 内容。已有 checkout 可执行：

```bash
git submodule update --init --recursive
```

需要在修改 Python 文件后立即使用新代码时，可将安装命令改为
`python -m pip install -v -e .`，保留相同的 MUSA 构建参数。修改 C++ 或 backend
文件后仍需重新构建。构建完成后运行[安装前检查](./01_version_environment.md#安装前检查)，
并在实际目标设备上验证 kernel。源码构建与预编译包可能带有不同的构建后缀，
请同时记录源码提交和运行环境。

### 历史版本（MUSA SDK 5.2.0 时代）

如果需要复现 MUSA SDK 5.2.0 时代的历史版本 `v0.1.8+musa.3`，请使用对应的包名和 TVM FFI 版本：

```bash
python -m pip install 'tilelang-musa==0.1.8+musa.3' \
  -i https://dl.mthreads.com/repo/api/pypi/pypi/simple
python -m pip install 'apache-tvm-ffi==0.1.9.post3' \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

```bash
python - <<'PY'
import tilelang
print("tilelang:", tilelang.__version__)
PY
```
