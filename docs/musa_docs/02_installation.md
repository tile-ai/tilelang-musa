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
   python -m pip install 'tilelang_musa==0.1.12+musa.2' \
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
python -m pip install 'tilelang_musa==0.1.12+musa.2'
python -m pip install 'apache-tvm-ffi==0.1.11.post1'
```

安装完成后仍通过 `import tilelang` 使用。

### 源码参考与版本渠道

TileLang-MUSA 源码仓库位于
[https://github.com/tile-ai/tilelang-musa](https://github.com/tile-ai/tilelang-musa)。
需要查看源码、提交记录或 release tag 时，可从仓库的
[Releases](https://github.com/tile-ai/tilelang-musa/releases) 页面选择
`v0.1.12+musa.1` 或 `v0.1.13+musa.1`。

本页前面的 pip 命令安装的是 MUSA SDK 配套发行版 `v0.1.12+musa.2`。它与开源
`v0.1.12+musa.1` 具有相同的基础 TileLang 版本，但包含额外 bugfix；开源
release 适合源码和测试对照，不能直接替代 MUSA SDK 配套包。Python 包版本和
GitHub release 版本需要分别核对完整的 `musa.*` patch 后缀。

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
