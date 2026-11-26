from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline, register_pipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)
from tilelang.musa.target import target_get_arch

TME_FLUSH_EXTERN = "tl::tme_flush"
TME_INTRINSIC_OPS = ("tl.tma_load", "tl.tma_load_im2col", "tl.tma_store")


def _extern_name(call: tirx.Call):
    op = getattr(call, "op", None)
    name = getattr(op, "name", None)
    if not isinstance(name, str) or not name.endswith("call_extern"):
        return None
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, tirx.StringImm):
        return str(first.value)
    return None


def _contains_tme_call(node) -> bool:
    found = False

    def visit(n):
        nonlocal found
        if found or not isinstance(n, tirx.Call):
            return None
        name = getattr(getattr(n, "op", None), "name", None)
        if name in TME_INTRINSIC_OPS:
            found = True
            return None
        extern = _extern_name(n)
        if extern is not None and extern.startswith("tl::tme_") and extern != TME_FLUSH_EXTERN:
            found = True
        return None

    tirx.stmt_functor.post_order_visit(node, visit)
    return found


def _contains_tme_flush(node) -> bool:
    found = False

    def visit(n):
        nonlocal found
        if found or not isinstance(n, tirx.Call):
            return None
        if _extern_name(n) == TME_FLUSH_EXTERN:
            found = True
        return None

    tirx.stmt_functor.post_order_visit(node, visit)
    return found


def _is_thread_extent(stmt) -> bool:
    return isinstance(stmt, tirx.AttrStmt) and isinstance(stmt.attr_key, str) and stmt.attr_key.startswith("thread_extent")


def _append_stmt(stmt, extra):
    if isinstance(stmt, tirx.SeqStmt):
        return tirx.SeqStmt(list(stmt.seq) + [extra])
    return tirx.SeqStmt([stmt, extra])


def _rebuild_container(node, body):
    if isinstance(node, tirx.AttrStmt):
        return tirx.AttrStmt(node.node, node.attr_key, node.value, body)
    if isinstance(node, tirx.SBlockRealize):
        block = node.block
        new_block = tirx.SBlock(
            iter_vars=block.iter_vars,
            reads=block.reads,
            writes=block.writes,
            body=body,
            init=block.init,
            alloc_buffers=block.alloc_buffers,
            match_buffers=block.match_buffers,
            annotations=block.annotations,
            name_hint=block.name_hint,
        )
        return tirx.SBlockRealize(node.iter_values, node.predicate, new_block)
    raise TypeError(f"Unsupported container statement: {type(node).__name__}")


def _append_innermost_thread_extent(node, stmt):
    if isinstance(node, tirx.SeqStmt):
        for idx in range(len(node.seq) - 1, -1, -1):
            rebuilt = _append_innermost_thread_extent(node.seq[idx], stmt)
            if rebuilt is not None:
                seq = list(node.seq)
                seq[idx] = rebuilt
                return tirx.SeqStmt(seq)
        return None
    if isinstance(node, tirx.SBlockRealize):
        rebuilt = _append_innermost_thread_extent(node.block.body, stmt)
        return None if rebuilt is None else _rebuild_container(node, rebuilt)
    if not _is_thread_extent(node):
        return None
    deeper = _append_innermost_thread_extent(node.body, stmt)
    if deeper is None:
        return _rebuild_container(node, _append_stmt(node.body, stmt))
    return _rebuild_container(node, deeper)


def add_tme_flush_cleanup(mod: IRModule) -> IRModule:
    """Append an unconditional ``tl::tme_flush()`` exit drain to every MP31 TME kernel."""
    out = IRModule(mod.functions, attrs=mod.attrs)
    for gv, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc) or not _contains_tme_call(func.body):
            continue
        if _contains_tme_flush(func.body):
            continue
        flush = tirx.Evaluate(tirx.call_extern("handle", TME_FLUSH_EXTERN))
        body = _append_innermost_thread_extent(func.body, flush)
        if body is None:
            body = _append_stmt(func.body, flush)
        out[gv] = tirx.PrimFunc(
            func.params,
            body,
            func.ret_type,
            func.buffer_map,
            func.attrs,
            func.span,
        )
    return out


def _allow_warp_specialized(target: Target) -> bool:
    if target_get_arch(target) != "mp_31":
        return False
    pass_ctx = tilelang.transform.get_pass_context()
    return not pass_ctx.config.get("tl.disable_warp_specialized", False)


def MUSAPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.VerifyBufferInit()(mod)
    if _allow_warp_specialized(target):
        mod = tilelang.musa.transform.ProducerConsumerWarpSpecialized()(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.musa.transform.LowerTMEIntrin()(mod)
    if target_get_arch(target) == "mp_31":
        mod = add_tme_flush_cleanup(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.musa.transform.LowerLDGSTG()(mod)
    # Run immediately before host/device splitting. Public fast-div markers
    # remain opaque until here so a fast_divmod pair can share its quotient;
    # host-side magic parameter setup must not be inlined into device code.
    mod = tilelang.musa.transform.LowerFastDivmod()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(
        enable_aggressive_merge=enable_aggressive_merge,
        disable_reuse=disable_reuse,
    )(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    if _allow_warp_specialized(target):
        mod = tilelang.musa.transform.LowerPartialThreadSync()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


register_pipeline(PassPipeline("musa", MUSAPassPipelineBody))
