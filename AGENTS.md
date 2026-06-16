# AGENTS.md

## AGENT 需要遵守的规则

- 使用中文回答问题, 代码和注释使用英文

## 项目概述

- 该项目对开源的 tilelang 项目进行修改, 加上对 MUSA 的适配, 并且只支持 MUSA 这一个 backend
- MUSA 是类似于 CUDA 的开发套件，适用于 Moore Thread 的 GPU, MUSA 努力和 CUDA API 都兼容，但是会有一些不兼容的地方
- 运行 Tilelang 程序会 JIT 编译生成 MUSA C 代码, 过程是 Tilelang -> TIR -> MUSA C
- TIR 是 TVM 项目开发的 AST 形式的 IR
- JIT 编译过程 IR 会经过很多 Pass, 这些 Pass 都在 src/transform 目录下
- 最终的 IR 经过 Codegen 生成 MUSA C 代码, Codegen 在 src/target/codegen_musa.cc 中
- MUSA C 代码会调用 tl 命名空间下的函数，这些函数都在 src/tl_templates/musa 目录下

## 构建,运行,测试命令

- 构建 Release 版本使用 `pip install -e . -v --no-build-isolation`
- 构建 Debug 版本使用 `pip install -e . -v --no-build-isolation -C cmake.build-type=Debug`
- 已存在 build 目录的情况下, 说明之前已经构建, 增量构建只需要使用 `cmake --build build`
- 执行 Tilelang 程序 `python example.py`
- Tilelang 程序里面通常会有一个 torch 写的 ref 程序, 做精度比较, 测试正确性直接执行 Tilelang 程序即可
- 打印生成的 MUSA C 代码使用 `print(kernel.get_kernel_source())`
- 打印中间的 TIR 设置 pass_config 如下
```py
pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
    tilelang.PassConfigKey.TL_DUMP_IR_DIR: "./dump_ir",
}
```

## 代码规范

- 编写 Tilelang Kernel 必须使用 Frontend v2 语法
- 修改之后检查下是否能简化冗余代码或逻辑

