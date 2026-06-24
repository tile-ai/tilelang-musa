/*
 * TMA lowering helpers split from copy.cc.
 * This file is included from copy.cc inside namespace tvm::tl.
 */

// Lowers copy to a bulk TMA (Tensor Memory Accelerator) transfer.
// Falls back to LowerNormalCopy if preconditions are not satisfied.
Stmt CopyNode::LowerBulkCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                             CopyInst copy_inst) const {
  ICHECK(copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_load = copy_inst == CopyInst::kBulkLoad;
  Buffer global_tensor = is_load ? src : dst;
  Buffer shared_tensor = is_load ? dst : src;
  Buffer shared_tensor_unmapped = shared_tensor;
  PrimExpr tma_barrier = GetTmaBarrier();
  Array<Range> global_range = is_load ? src_range : dst_range;
  Array<Range> shared_range = is_load ? dst_range : src_range;
  PrimExpr total_copy_bytes = shared_tensor->dtype.bytes();
  for (const auto &range : shared_range) {
    total_copy_bytes *= range->extent;
  }
  total_copy_bytes = analyzer->Simplify(total_copy_bytes);
  auto lower_barriered_normal_fallback = [&]() -> Stmt {
    bool needs_external_arrive =
        TargetIsMusa(T.target) && is_load && GetIsTmaCopy() &&
        annotations.Get("barrier").has_value();
    if (!needs_external_arrive) {
      return LowerNormalCopy(T, analyzer);
    }

    auto fallback_op = tvm::ffi::make_object<CopyNode>(*this);
    auto fallback_annotations = fallback_op->annotations;
    // The fallback must be complete before releasing the WS mbarrier. Avoid
    // re-injecting a bare force_async_copy here because MUSA lacks a
    // cp.async-mbarrier coupling instruction.
    fallback_annotations.erase("force_async_copy");
    fallback_annotations.erase("is_async_copy");
    fallback_annotations.erase("force_cp_async");
    fallback_annotations.erase(attr::kAsyncCopyNoImplicitCommitWait);
    fallback_op->annotations = fallback_annotations;

    Stmt copy = fallback_op->LowerNormalCopy(T, analyzer);
    Stmt arrive =
        Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(),
                      {GetTmaBarrier()}));
    return SeqStmt({copy, arrive});
  };
  // TMA bulk copy cannot support a non-swizzled global layout, will be fallback
  // to normal copy
  if (T.layout_map.count(global_tensor)) {
    LOG(WARNING) << "TMA bulk copy cannot support a non-swizzled global "
                    "layout, fallback to normal copy.";
    return lower_barriered_normal_fallback();
  }

  // linear layout must be computed before remapping
  auto linear_layout = ComputeLinearLayout(shared_tensor);

  Array<PrimExpr> shared_indices;
  for (auto r : shared_range)
    shared_indices.push_back(r->min);
  std::vector<PrimExpr> shared_strides;
  PrimExpr shared_stride = 1;
  for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
    auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
    shared_strides.insert(shared_strides.begin(), shared_stride);
    shared_stride *= s;
  }

  Array<PrimExpr> global_indices;
  for (auto r : global_range) {
    global_indices.push_back(r->min);
  }
  std::vector<PrimExpr> global_strides;
  PrimExpr global_stride = 1;
  for (size_t i = 0; i < global_tensor->shape.size(); i++) {
    auto s = global_tensor->shape[global_tensor->shape.size() - i - 1];
    global_strides.insert(global_strides.begin(), global_stride);
    global_stride *= s;
  }

  ICHECK(shared_strides.size() == shared_indices.size())
      << "shared_strides.size() != shared_indices.size()"
      << shared_strides.size() << " " << shared_indices.size();
  PrimExpr shared_offset = 0;
  for (size_t i = 0; i < shared_indices.size(); i++) {
    shared_offset += shared_indices[i] * shared_strides[i];
  }
  PrimExpr global_offset = 0;
  for (size_t i = 0; i < global_indices.size(); i++) {
    global_offset += global_indices[i] * global_strides[i];
  }

  TMADesc desc;
  // Verify copy rank
  desc.rank = global_tensor->shape.size();
  ICHECK(desc.rank >= 1 && desc.rank <= 5) << desc.rank;

  // Verify datatype
  ICHECK(global_tensor->dtype == shared_tensor->dtype)
      << "Copy between buffer " << global_tensor->name << " and "
      << shared_tensor->name << " with different data type "
      << global_tensor->dtype << " and " << shared_tensor->dtype;

  desc.data_type = TargetIsMusa(T.target)
                       ? to_MUtensorDescriptorDataType(global_tensor->dtype)
                       : to_CUtensorMapDataType(global_tensor->dtype);

  // Global Tensor Shape and Stride
  desc.global_addr = global_tensor->data;
  desc.global_shape = ReverseArray(global_tensor->shape);
  Array<PrimExpr> global_coords =
      ReverseArray(global_range.Map([](Range r) { return r->min; }));
  if (!global_tensor->strides.empty()) {
    desc.global_stride = ReverseArray(global_tensor->strides);
  } else {
    // Create stride from shape
    PrimExpr stride = 1;
    desc.global_stride.reserve(desc.rank);
    for (size_t i = 0; i < desc.rank; i++) {
      desc.global_stride.push_back(stride);
      stride *= desc.global_shape[i];
    }
  }
  // The first stride element should be 1
  ICHECK(is_one(desc.global_stride[0])) << desc.global_stride;
  // Make global stride in bytes
  auto stride_dtype =
      TargetIsMusa(T.target) ? DataType::UInt(64) : DataType::Int(64);
  desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
    return cast(stride_dtype, e) * global_tensor->dtype.bytes();
  });
  for (size_t i{1}; i < desc.global_stride.size(); i++) {
    auto stride = desc.global_stride[i].as<IntImmNode>();
    if (stride != nullptr) {
      // otherwise, the stride is symbolic, we need to check in future with
      // assumptions
      if (stride->value % 16 != 0 || stride->value >= (1ULL << 40)) {
        LOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                     << desc.global_stride[i] << ", fallback to normal copy.";
        return lower_barriered_normal_fallback();
      }
    }
  }

  // Smem Box
  // check smem range and global range is legal
  auto s_range_idx = 0;
  for (size_t i = 0; i < global_range.size(); i++) {
    auto g_range = global_range[i];
    if (is_one(g_range->extent)) {
      continue;
    }
    // skip one range if it is 1
    // in case of global range is [128, 64], while shared range is [1, 128, 64]
    // A_shared[0, :, :].
    while (s_range_idx < shared_range.size() &&
           is_one(shared_range[s_range_idx]->extent)) {
      s_range_idx++;
    }
    if (s_range_idx >= shared_range.size()) {
      LOG(FATAL) << "TMA bulk copy cannot support a global range of "
                 << global_range << ", shared_range " << shared_range;
    }
    auto s_range = shared_range[s_range_idx];
    s_range_idx++;

    auto g_extent = analyzer->Simplify(g_range->extent);
    auto s_extent = analyzer->Simplify(s_range->extent);
    ICHECK(analyzer->CanProveEqual(g_extent, s_extent))
        << global_tensor->name << "[" << i << "] is illegal, "
        << global_tensor->name << "[" << i << "] = " << g_range->extent << ", "
        << shared_tensor->name << "[" << s_range_idx
        << "] = " << s_range->extent;
  }
  // TODO(lei): find a much smarter way to deduce smem box dim
  // instead of using global_range
  desc.smem_box =
      ReverseArray(global_range.Map([](Range r) { return r->extent; }));

  desc.smem_stride = Array<PrimExpr>(desc.rank, PrimExpr(1));
  // L2 & OOB
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // Detect smem layout
  // Shared memory swizzling is crucial for TMA performance
  // It determines how data is arranged in shared memory banks to minimize bank
  // conflicts Different swizzle patterns (32B, 64B, 128B) offer different
  // trade-offs between access efficiency and memory usage
  desc.interleave = TargetIsMusa(T.target)
                        ? static_cast<int>(MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE)
                        : static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  desc.swizzle = TargetIsMusa(T.target)
                     ? static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_NONE)
                     : static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  Layout shared_layout;
  auto active_layout = GetActiveLayout(T, shared_tensor);
  if (active_layout.has_value()) {
    shared_layout = active_layout.value();
  }
  if (T.buffer_remap.count(shared_tensor)) {
    shared_tensor = T.buffer_remap.at(shared_tensor);
  }
  bool musa_force_swizzle_none = false;
  if (!shared_layout.defined()) {
    desc.swizzle = TargetIsMusa(T.target)
                       ? static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_NONE)
                       : static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else if (StructuralEqual()(shared_layout, linear_layout)) {
    desc.swizzle = TargetIsMusa(T.target)
                       ? static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_NONE)
                       : static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
    musa_force_swizzle_none = TargetIsMusa(T.target);
  } else if (TargetIsMusa(T.target) && IsMatrixLinearLayout(shared_layout)) {
    desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_NONE);
    musa_force_swizzle_none = true;
  } else if (!TargetIsMusa(T.target)) {
    if (shared_layout->InputDim() < 2) {
      LOG(WARNING) << "TMA bulk copy cannot support shared layout with input "
                   << "dimension " << shared_layout->InputDim()
                   << ", fallback to normal copy.";
      return lower_barriered_normal_fallback();
    }
    const int ndim = static_cast<int>(shared_layout->InputDim());
    auto stride = as_const_int(shared_layout->InputShape()[ndim - 2]);
    auto continuous = as_const_int(shared_layout->InputShape()[ndim - 1]);
    ICHECK(stride != nullptr && continuous != nullptr);
    // We also need to check if the shape satisfies the following doc:
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    SwizzleMode swizzle_mode =
        DetectSwizzleMode(shared_layout, shared_tensor_unmapped);
    if (swizzle_mode == SwizzleMode::kQuarter) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (swizzle_mode == SwizzleMode::kHalf) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (swizzle_mode == SwizzleMode::kFull) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else if (StructuralEqual()(
                   shared_layout,
                   makeGemmABLayoutPadded(*stride, *continuous,
                                          shared_tensor->dtype.bits()))) {
      LOG(WARNING) << "Bulk copy cannot support a padded layout for src: "
                   << src->name << ", dst: " << dst->name
                   << ", fallback to normal copy";
      return lower_barriered_normal_fallback();
    } else {
      LOG(WARNING) << "Came across unsupported swizzle layout for src: "
                   << src->name << ", dst: " << dst->name
                   << ", fallback to normal copy";
      return lower_barriered_normal_fallback();
    }
  }

  auto get_k_major = [&](const Buffer &buf) -> int {
    auto k_major_opt = GetLayoutKMajor(T, buf);
    if (k_major_opt.has_value()) {
      return k_major_opt.value() ? 1 : 0;
    }
    return -1;
  };
  // set swizzle granularity by elem_bytes and k_major
  if (TargetIsMusa(T.target) && !musa_force_swizzle_none) {
    int elem_bytes = shared_tensor->dtype.bytes();
    int is_k_major = get_k_major(shared_tensor);
    if (is_k_major < 0 && !shared_tensor.same_as(shared_tensor_unmapped)) {
      is_k_major = get_k_major(shared_tensor_unmapped);
    }
    if (elem_bytes == 1) {
      desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B16);
    } else if (elem_bytes == 2) {
      if (is_k_major == 1) {
        desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B16);
      } else if (is_k_major == 0) {
        desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B32);
      } else {
        LOG(INFO) << src->name << " use elem_bytes " << elem_bytes
                  << ", unknown matrix layout for swizzle";
      }
    } else if (elem_bytes == 4) {
      if (is_k_major == 1) {
        desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B16);
      } else if (is_k_major == 0) {
        desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B64);
      } else {
        LOG(INFO) << src->name << " use elem_bytes " << elem_bytes
                  << ", unknown matrix layout for swizzle";
      }
    } else {
      LOG(WARNING) << src->name << " use elem_bytes " << elem_bytes
                   << ", swizzle not set";
    }
  }

  int split_count = 1;
  PrimExpr split_stride = 0;
  PrimExpr split_size_expr = 0;
  if (TargetIsMusa(T.target) && is_load) {
    auto sqmma_opt = GetLayoutSQMMA(T, shared_tensor);
    auto k_major_opt = GetLayoutKMajor(T, shared_tensor);
    if (!sqmma_opt.has_value() &&
        !shared_tensor.same_as(shared_tensor_unmapped)) {
      sqmma_opt = GetLayoutSQMMA(T, shared_tensor_unmapped);
    }
    if (!k_major_opt.has_value() &&
        !shared_tensor.same_as(shared_tensor_unmapped)) {
      k_major_opt = GetLayoutKMajor(T, shared_tensor_unmapped);
    }
    bool is_musa_sqmma_tma_load =
        src.scope() == "global" &&
        (dst.scope() == "shared" || dst.scope() == "shared.dyn") &&
        k_major_opt.has_value() && sqmma_opt.value_or(false);
    if (is_musa_sqmma_tma_load) {
      auto n_dim = as_const_int(desc.smem_box[0]);
      auto make_split_stride = [&](PrimExpr split_size) {
        PrimExpr stride = split_size;
        for (size_t i = 1; i < desc.smem_box.size(); ++i) {
          stride *= desc.smem_box[i];
        }
        return analyzer->Simplify(stride);
      };
      bool is_k_major = k_major_opt.value();
      if (!is_k_major) {
        int sqmma_inst_split =
            GetLayoutSqmmaInstSplit(T, shared_tensor).value_or(-1);
        if (sqmma_inst_split < 0 &&
            !shared_tensor.same_as(shared_tensor_unmapped)) {
          sqmma_inst_split =
              GetLayoutSqmmaInstSplit(T, shared_tensor_unmapped).value_or(-1);
        }
        if (sqmma_inst_split > 0 && n_dim != nullptr &&
            (*n_dim) > sqmma_inst_split && ((*n_dim) % sqmma_inst_split) == 0) {
          split_count = (*n_dim) / sqmma_inst_split;
          split_size_expr = IntImm(DataType::Int(32), sqmma_inst_split);
          split_stride = make_split_stride(split_size_expr);
          desc.smem_box.Set(0, split_size_expr);
        }
      } else {
        int elem_bytes = shared_tensor->dtype.bytes();
        int max_inner_elems = elem_bytes > 0 ? 256 / elem_bytes : 0;
        if (max_inner_elems > 0 && n_dim != nullptr &&
            (*n_dim) * elem_bytes > 256 && ((*n_dim) % max_inner_elems) == 0) {
          split_count = (*n_dim) / max_inner_elems;
          split_size_expr = IntImm(DataType::Int(32), max_inner_elems);
          split_stride = make_split_stride(split_size_expr);
          desc.smem_box.Set(0, split_size_expr);
        }
      }
    }
  }

  auto inner_box_dim = as_const_int(desc.smem_box[0]);
  if (inner_box_dim == nullptr) {
    LOG(WARNING) << "inner_box_dim " << desc.smem_box[0]
                 << " can only be a constant integer for TMA bulk copy, "
                    "fallback to normal copy";
    return lower_barriered_normal_fallback();
  }
  int instruction_dim = *inner_box_dim;
  if (!TargetIsMusa(T.target)) {
    if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B)) {
      instruction_dim = 64 / src->dtype.bytes();
    } else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B)) {
      instruction_dim = 128 / src->dtype.bytes();
    }
  }
  if (!TargetIsMusa(T.target) && instruction_dim > 256) {
    // smem_box dim must be in [0, 256]
    // if is 512, we need to split the copy into two parts
    ICHECK((*inner_box_dim) % 256 == 0)
        << "inner_box_dim: " << *inner_box_dim << " is not divisible by 256";
    instruction_dim = 256;
  }
  ICHECK((*inner_box_dim) % instruction_dim == 0)
      << "inner_box_dim: " << *inner_box_dim
      << " is not divisible by instruction_dim: " << instruction_dim;
  desc.smem_box.Set(0, PrimExpr(instruction_dim));

  int inner_box_dim_ = instruction_dim * shared_tensor->dtype.bytes();

  // Check inner_box_dim_ for each swizzle type in a cleaner way
  struct SwizzleCheck {
    int swizzle;
    int max_dim;
  };
  static const std::vector<SwizzleCheck> swizzle_checks = {
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B), 32},
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B), 64},
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B), 128},
  };
  if (!TargetIsMusa(T.target)) {
    for (const auto &check : swizzle_checks) {
      if (desc.swizzle == check.swizzle && inner_box_dim_ > check.max_dim) {
        LOG(WARNING) << "TMA bulk copy cannot support a swizzled global layout "
                        "with inner_box_dim_ > "
                     << check.max_dim << ", will be fallback to normal copy";
        return lower_barriered_normal_fallback();
      }
    }
  }

  if (TargetIsMusa(T.target)) {
    desc.global_shape = desc.global_shape.Map(
        [](PrimExpr e) { return cast(DataType::UInt(64), e); });
  }

  Call create_descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());

  // For TMA loads, allocate mbarrier(s) for synchronous semantics.
  // Determine the mbarrier handle for TMA loads.
  // T.tma_copy(): requires user-provided barrier
  // T.copy(): allocates internal mbarrier via AllocMBarrier
  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  bool is_cluster_barrier = false;
  if (is_load) {
    if (auto user_barrier = annotations.Get("barrier")) {
      mbar_handle = Downcast<PrimExpr>(user_barrier.value());
      barrier_base_id = 0;
      if (auto bl = mbar_handle.as<BufferLoadNode>()) {
        is_cluster_barrier = bl->buffer.scope() == "shared.cluster_barrier";
      }
    } else if (GetIsTmaCopy()) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx]).";
    } else if (T.AllocMBarrier) {
      // Internal mbarrier (T.copy()): allocate a single barrier slot.
      // Pipeline buffer versioning expands it per stage when needed.
      barrier_base_id = T.AllocMBarrier(1);
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(T.mbarrier_buffer->value(), {mbar_idx});
    }
  }
  PrimExpr load_barrier_arg = barrier_base_id >= 0 ? mbar_handle : tma_barrier;
  auto op = is_load ? tma_load() : tma_store();

  Stmt tma_copy;
  PrimExpr total_elements = 1;
  for (auto e : desc.smem_box)
    total_elements *= e;

  if (TargetIsMusa(T.target) && split_count > 1) {
    Array<PrimExpr> base_args;
    base_args.reserve(desc.rank + 4);
    base_args.push_back(create_descriptor);
    if (is_load)
      base_args.push_back(load_barrier_arg);

    auto emit_tma_copy = [&](PrimExpr shared_addr,
                             const Array<PrimExpr> &coords) -> Stmt {
      Array<PrimExpr> call_args = base_args;
      call_args.push_back(shared_addr);
      for (auto coord : coords)
        call_args.push_back(coord);
      int need_reduce = 0;
      if (!is_load)
        call_args.push_back(need_reduce);
      call_args.push_back(GetEvictionPolicy());
      call_args.push_back(GetInnerCachePolicy());
      call_args.push_back(GetOuterCachePolicy());
      return Evaluate(Call(DataType::Handle(), op, call_args));
    };

    auto make_inner_copy = [&](PrimExpr base_shared_offset,
                               const Array<PrimExpr> &base_coords) -> Stmt {
      if ((*inner_box_dim) != instruction_dim) {
        Var loop_var("i");
        int loop_extent = (*inner_box_dim) / instruction_dim;
        Array<PrimExpr> coords = base_coords;
        coords.Set(0, coords[0] + instruction_dim * loop_var);
        PrimExpr shared_addr = shared_tensor.access_ptr(
            is_load ? 2 : 1, DataType::Handle(), 1,
            base_shared_offset + total_elements * loop_var, total_elements);
        return For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                   emit_tma_copy(shared_addr, coords));
      } else {
        PrimExpr shared_addr =
            shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(), 1,
                                     base_shared_offset, total_elements);
        return emit_tma_copy(shared_addr, base_coords);
      }
    };

    Var split_var("s");
    Array<PrimExpr> coords = global_coords;
    coords.Set(0, coords[0] + split_size_expr * split_var);
    PrimExpr base_shared_offset = shared_offset + split_stride * split_var;
    tma_copy = For(split_var, 0, split_count, ForKind::kUnrolled,
                   make_inner_copy(base_shared_offset, coords));
  } else {
    Array<PrimExpr> args;
    args.reserve(desc.rank + 4);
    args.push_back(create_descriptor);
    if (is_load)
      args.push_back(load_barrier_arg);
    if ((*inner_box_dim) != instruction_dim) {
      Var loop_var("i");
      int loop_extent = (*inner_box_dim) / instruction_dim;
      PrimExpr shared_addr = shared_tensor.access_ptr(
          is_load ? 2 : 1, DataType::Handle(), 1,
          shared_offset + total_elements * loop_var, total_elements);
      args.push_back(shared_addr);
      global_coords.Set(0, global_coords[0] + instruction_dim * loop_var);
      for (auto coord : global_coords)
        args.push_back(coord);
      int need_reduce = 0;
      if (!is_load)
        args.push_back(need_reduce);
      args.push_back(GetEvictionPolicy());
      args.push_back(GetInnerCachePolicy());
      args.push_back(GetOuterCachePolicy());
      tma_copy = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                     Evaluate(Call(DataType::Handle(), op, args)));
    } else {
      PrimExpr shared_addr =
          shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(), 1,
                                   shared_offset, total_elements);
      args.push_back(shared_addr);
      for (auto coord : global_coords)
        args.push_back(coord);
      int need_reduce = 0;
      if (!is_load)
        args.push_back(need_reduce);
      args.push_back(GetEvictionPolicy());
      args.push_back(GetInnerCachePolicy());
      args.push_back(GetOuterCachePolicy());
      tma_copy = Evaluate(Call(DataType::Handle(), op, args));
    }
  }

  // Bulk TMA stores participate in the async store group mechanism, so they
  // must be committed. T.copy() keeps synchronous semantics by waiting here;
  // T.tma_copy() leaves the wait to the user for explicit batching.
  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    if (!GetIsTmaCopy()) {
      seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                  {IntImm(DataType::Int(32), 0)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  // For TMA loads with inline mbarrier: emit expect_tx before tma_load
  // (inside thread-gated block), and wait_parity after (all threads).
  // The producer is annotated with the shared buffer so PipelinePlanning can
  // detect it as a copy stage and schedule it at pipeline stage 0.
  if (is_load && barrier_base_id >= 0) {
    // Compute total bytes for all TMA sub-copies in this operation
    PrimExpr total_bytes;
    if ((*inner_box_dim) != instruction_dim) {
      int loop_extent = (*inner_box_dim) / instruction_dim;
      total_bytes = total_elements * loop_extent * shared_tensor->dtype.bytes();
    } else {
      total_bytes = total_elements * shared_tensor->dtype.bytes();
    }
    if (TargetIsMusa(T.target) && split_count > 1) {
      total_bytes *= split_count;
    }
    total_bytes = analyzer->Simplify(total_bytes);

    Stmt barrier_before_tma_stmt;
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (GetIsTmaCopy()) {
      // T.tma_copy(): only expect_tx (no arrive). User must call
      // T.barrier_arrive() explicitly. This allows multiple tma_copy operations
      // to share a single arrive.
      if (is_cluster_barrier) {
        // For cluster barriers in 2CTA mode: all CTAs' TMA arrivals go to
        // CTA 0's barrier (via tma_load_2sm peer-bit clearing). So expect_tx
        // must account for ALL CTAs' bytes and only execute on CTA 0.
        PrimExpr cluster_total_bytes =
            total_bytes * IntImm(DataType::Int(32), T.cluster_size);
        Stmt expect_stmt =
            Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                          {mbar_handle, cluster_total_bytes}));
        PrimExpr rank = Call(DataType::Int(32), block_rank_in_cluster(), {});
        barrier_before_tma_stmt =
            IfThenElse(EQ(rank, IntImm(DataType::Int(32), 0)), expect_stmt);
      } else {
        barrier_before_tma_stmt =
            Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                          {mbar_handle, total_bytes}));
      }
      // When emit_arrive is set (by InjectSoftwarePipeline for pipeline-level
      // barrier management), also emit arrive inside the thread-0 guard.
      if (auto emit_arrive_val = annotations.Get("emit_arrive")) {
        if (Downcast<IntImm>(emit_arrive_val.value())->value != 0) {
          barrier_after_tma_stmt =
              Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(),
                            {mbar_handle}));
        }
      }
    } else {
      // T.copy() with TMA: keep expect_tx and arrive as separate control ops.
      // This lets downstream WS/barrier passes reason about the arrival
      // domain explicitly when TMA shares a stage barrier with cp.async.
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      barrier_after_tma_stmt = Evaluate(Call(
          DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));
    }

    Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy};
    if (barrier_after_tma_stmt.defined()) {
      producer_seq.push_back(barrier_after_tma_stmt.value());
    }

    // Thread-gated block: expect_tx + tma_load (+ optional arrive)
    Stmt producer = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                               SeqStmt(producer_seq));

    // tma_copy (from T.tma_copy()) is fire-and-forget: only emit the
    // producer (expect_tx + tma_load). The user manages synchronization
    // (arrive + wait) explicitly.
    if (GetIsTmaCopy()) {
      return producer;
    }

    // For T.copy() with TMA: emit producer + wait pair so the pipeline/WS
    // passes can split them into different stages.
    Stmt wait_stmt =
        Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                      {mbar_handle, GetCopyMbarPhaseExpr(annotations, T)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy =
      IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent), tma_copy);

  return tma_copy;
}

Stmt CopyNode::LowerBulkCopy1D(const LowerArgs &T, arith::Analyzer *analyzer,
                               CopyInst copy_inst) const {
  ICHECK(copy_inst == CopyInst::kBulkLoad1D ||
         copy_inst == CopyInst::kBulkStore1D);

  // Add 1D TMA copy when the global and shared memory is contiguous
  // Check if shared_tensor->name is present in T.buffer_var_gemm
  // (Array<PrimExpr>) to avoid use 1D TMA copy for swizzled layout
  bool is_load = copy_inst == CopyInst::kBulkLoad1D;
  auto shared_range = is_load ? dst_range : src_range;
  auto global_range = is_load ? src_range : dst_range;
  auto shared_tensor = is_load ? dst : src;
  auto global_tensor = is_load ? src : dst;
  PrimExpr tma_barrier = GetTmaBarrier();

  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }

  std::vector<PrimExpr> shared_strides;
  PrimExpr shared_stride = 1;
  for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
    auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
    shared_strides.insert(shared_strides.begin(), shared_stride);
    shared_stride *= s;
  }

  Array<PrimExpr> shared_indices;
  for (auto r : shared_range)
    shared_indices.push_back(r->min);

  Array<PrimExpr> global_indices;
  for (auto r : global_range) {
    global_indices.push_back(r->min);
  }
  std::vector<PrimExpr> global_strides;
  PrimExpr global_stride = 1;
  for (size_t i = 0; i < global_tensor->shape.size(); i++) {
    auto s = global_tensor->shape[global_tensor->shape.size() - i - 1];
    global_strides.insert(global_strides.begin(), global_stride);
    global_stride *= s;
  }

  PrimExpr global_offset = 0;
  for (size_t i = 0; i < global_indices.size(); i++) {
    global_offset += global_indices[i] * global_strides[i];
  }

  PrimExpr shared_offset = 0;
  for (size_t i = 0; i < shared_indices.size(); i++) {
    shared_offset += shared_indices[i] * shared_strides[i];
  }

  PrimExpr elements = analyzer->Simplify(shared_elements);
  PrimExpr shared_addr = shared_tensor.access_ptr(
      is_load ? 2 : 1, DataType::Handle(), 1, shared_offset, elements);
  PrimExpr global_addr = global_tensor.access_ptr(
      is_load ? 1 : 2, DataType::Handle(), 1, global_offset, elements);
  PrimExpr total_bytes =
      analyzer->Simplify(elements * shared_tensor->dtype.bytes());
  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  if (is_load) {
    if (auto user_barrier = annotations.Get("barrier")) {
      mbar_handle = Downcast<PrimExpr>(user_barrier.value());
      barrier_base_id = 0;
    } else if (GetIsTmaCopy()) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx]).";
    } else if (T.AllocMBarrier) {
      // Internal mbarrier (T.copy()): allocate a single barrier slot.
      // Pipeline buffer versioning expands it per stage when needed.
      barrier_base_id = T.AllocMBarrier(1);
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(T.mbarrier_buffer->value(), {mbar_idx});
    }
  }
  PrimExpr load_barrier_arg = barrier_base_id >= 0 ? mbar_handle : tma_barrier;

  Stmt tma_copy;
  if (is_load) {
    tma_copy = Evaluate(Call(DataType::Handle(), tma_load(),
                             {shared_addr, global_addr, load_barrier_arg,
                              total_bytes, GetEvictionPolicy(),
                              GetInnerCachePolicy(), GetOuterCachePolicy()}));
  } else {
    int need_reduce = 0;
    tma_copy = Evaluate(Call(DataType::Handle(), tma_store(),
                             {global_addr, shared_addr, total_bytes,
                              need_reduce, GetEvictionPolicy(),
                              GetInnerCachePolicy(), GetOuterCachePolicy()}));
  }

  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    if (!GetIsTmaCopy()) {
      seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                  {IntImm(DataType::Int(32), 0)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  // For 1D TMA loads with inline mbarrier: emit expect_tx + tma_load
  // (inside thread-gated block), and wait_parity after (all threads).
  if (is_load && barrier_base_id >= 0) {
    Stmt barrier_before_tma_stmt;
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (GetIsTmaCopy()) {
      // T.tma_copy(): only expect_tx (no arrive). User must call
      // T.barrier_arrive() explicitly. This allows multiple tma_copy operations
      // to share a single arrive.
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
    } else {
      // T.copy() with TMA: keep expect_tx and arrive as separate control ops.
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      barrier_after_tma_stmt = Evaluate(Call(
          DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));
    }

    Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy};
    if (barrier_after_tma_stmt.defined()) {
      producer_seq.push_back(barrier_after_tma_stmt.value());
    }

    Stmt producer = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                               SeqStmt(producer_seq));

    // tma_copy (from T.tma_copy()) is fire-and-forget: only emit the
    // producer (expect_tx + tma_load). The user manages synchronization
    // (arrive + wait) explicitly.
    if (GetIsTmaCopy()) {
      return producer;
    }

    // For T.copy() with TMA: emit producer + wait pair so the pipeline/WS
    // passes can split them into different stages.
    Stmt wait_stmt =
        Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                      {mbar_handle, GetCopyMbarPhaseExpr(annotations, T)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy =
      IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent), tma_copy);
  return tma_copy;
}
// Encodes the TMA descriptor into an array of PrimExpr for
// create_tma_descriptor().
Array<PrimExpr> TMADesc::EncodeCallArgs() const {
  Array<PrimExpr> args;
  args.reserve(rank * 4 + 7);

  args.push_back(data_type);
  args.push_back(static_cast<int>(rank));
  args.push_back(global_addr);
  for (auto e : global_shape)
    args.push_back(e);
  for (auto e : global_stride)
    args.push_back(e);
  for (auto e : smem_box)
    args.push_back(e);
  for (auto e : smem_stride)
    args.push_back(e);
  args.push_back(interleave);
  args.push_back(swizzle);
  args.push_back(l2_promotion);
  args.push_back(oob_fill);

  return args;
}

// Constructs a Conv2DIm2ColOp node from call arguments.
// args: src, dst, nhw_step, c_step, kernel, stride, dilation, padding,
// eviction_policy
Conv2DIm2ColOp::Conv2DIm2ColOp(Array<PrimExpr> args,
                               Map<String, ObjectRef> annotations) {
  ObjectPtr<Conv2DIm2ColOpNode> node =
      tvm::ffi::make_object<Conv2DIm2ColOpNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src_ = node->srcRegion_->buffer;
  node->dst_ = node->dstRegion_->buffer;
  node->nhw_step_ = args[2];
  node->c_step_ = args[3];
  node->kernel_ = args[4].as<IntImm>().value()->value;
  node->stride_ = args[5].as<IntImm>().value()->value;
  node->dilation_ = args[6].as<IntImm>().value()->value;
  node->padding_ = args[7].as<IntImm>().value()->value;
  node->eviction_policy_ = args[8].as<IntImm>().value()->value;
  node->annotations_ = annotations;
  data_ = std::move(node);
}

// Creates a shallow copy of this Conv2DIm2ColOpNode.
TileOperator Conv2DIm2ColOpNode::Clone() const {
  auto op = tvm::ffi::make_object<Conv2DIm2ColOpNode>(*this);
  return Conv2DIm2ColOp(op);
}

// Lowers Conv2D im2col into a TMA-backed PTX sequence for Hopper.
Stmt Conv2DIm2ColOpNode::Lower(const LowerArgs &T,
                               arith::Analyzer *analyzer) const {
  ICHECK(TargetIsHopper(T.target));
  ICHECK(IsGlobalBuffer(src_) && IsSharedBuffer(dst_));
  ICHECK(src_->shape.size() == 4);
  ICHECK(src_->dtype == dst_->dtype);

  // Use dstRegion_ to derive tile dimensions and shared memory offset.
  // dstRegion_ always has the correct ranges regardless of whether MVB
  // added a leading stage dimension to the buffer — the last two ranges
  // give the tile (pixel, channel) extents and mins.
  size_t ndim = dstRegion_->region.size();
  ICHECK(ndim >= 2) << "im2col dstRegion must have at least 2 dims";
  Layout shared_layout;
  if (T.layout_map.count(dst_)) {
    shared_layout = T.layout_map[dst_];
  }

  TMAIm2ColDesc desc;
  desc.rank = src_->shape.size();
  desc.data_type = TargetIsMusa(T.target)
                       ? to_MUtensorDescriptorDataType(src_->dtype)
                       : to_CUtensorMapDataType(src_->dtype);
  desc.global_addr = src_->data;
  desc.global_shape = ReverseArray(src_->shape);

  if (!src_->strides.empty()) {
    desc.global_stride = ReverseArray(src_->strides);
  } else {
    // Create stride from shape
    PrimExpr stride = 1;
    desc.global_stride.reserve(desc.rank);
    for (size_t i = 0; i < desc.rank; i++) {
      desc.global_stride.push_back(stride);
      stride *= desc.global_shape[i];
    }
  }
  // The first stride element should be 1
  ICHECK(is_one(desc.global_stride[0])) << desc.global_stride;
  // Make global stride in bytes
  desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
    return cast(DataType::Int(64), e) * src_->dtype.bytes();
  });
  desc.elem_stride = {1, stride_, stride_, 1};
  desc.lower_corner = {-padding_, -padding_};
  desc.upper_corner = {-padding_, -padding_};
  desc.smem_box_pixel =
      Downcast<IntImm>(dstRegion_->region[ndim - 2]->extent)->value;
  desc.smem_box_channel =
      Downcast<IntImm>(dstRegion_->region[ndim - 1]->extent)->value;
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  if (!shared_layout.defined()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    ICHECK(shared_layout->InputDim() >= 2) << "Cannot detect TMA layout.";
    auto swizzle_mode = DetectSwizzleMode(shared_layout, dst_);
    if (swizzle_mode == SwizzleMode::kQuarter) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (swizzle_mode == SwizzleMode::kHalf) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (swizzle_mode == SwizzleMode::kFull) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else {
      LOG(FATAL) << "Cannot detect TMA layout.";
    }
  }

  Call create_desc = Call(DataType::Handle(), create_tma_im2col_descriptor(),
                          desc.EncodeCallArgs());

  Array<PrimExpr> global_coords; // c, w, h, n
  Array<PrimExpr> image_offset;  // w, h
  global_coords.reserve(desc.rank);

  ICHECK(analyzer->CanProveEqual(
      FloorMod(desc.global_shape[0], desc.smem_box_channel), 0))
      << "Currently can only support divisible channel case";

  global_coords.push_back(
      FloorMod(c_step_ * desc.smem_box_channel, desc.global_shape[0]));
  image_offset.push_back(
      dilation_ *
      FloorMod(FloorDiv(c_step_ * desc.smem_box_channel, desc.global_shape[0]),
               kernel_));
  image_offset.push_back(dilation_ * FloorDiv(c_step_ * desc.smem_box_channel,
                                              desc.global_shape[0] * kernel_));

  PrimExpr h_dim =
      FloorDiv(src_->shape[1] + 2 * padding_ - (kernel_ - 1) * dilation_ - 1,
               stride_) +
      1;
  PrimExpr w_dim =
      FloorDiv(src_->shape[2] + 2 * padding_ - (kernel_ - 1) * dilation_ - 1,
               stride_) +
      1;
  global_coords.push_back(
      stride_ * FloorMod(nhw_step_ * desc.smem_box_pixel, w_dim) - padding_);
  global_coords.push_back(
      stride_ *
          FloorMod(FloorDiv(nhw_step_ * desc.smem_box_pixel, w_dim), h_dim) -
      padding_);
  global_coords.push_back(
      FloorDiv(nhw_step_ * desc.smem_box_pixel, w_dim * h_dim));

  // Allocate mbarrier(s) for TMA im2col load synchronization,
  // matching the protocol used by regular TMA loads.
  // If a barrier was provided by the WS pass (via annotation), use it directly.
  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  if (auto user_barrier = annotations_.Get("barrier")) {
    // WS pass provided a barrier: use it without allocating a new one.
    mbar_handle = Downcast<PrimExpr>(user_barrier.value());
    barrier_base_id = 0;
  } else if (T.AllocMBarrier) {
    // Allocate a single barrier slot; pipeline buffer versioning expands it
    // per stage when needed.
    barrier_base_id = T.AllocMBarrier(1);
    PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
    mbar_handle = BufferLoad(T.mbarrier_buffer->value(), {mbar_idx});
  }

  Array<PrimExpr> args;
  args.reserve(desc.rank * 2 + 2);
  args.push_back(create_desc);
  args.push_back(barrier_base_id >= 0 ? mbar_handle : PrimExpr(0));
  auto dst_buffer = T.buffer_remap.count(dst_) ? T.buffer_remap[dst_] : dst_;
  // Compute flat element offset from dstRegion_ mins and buffer strides.
  // For a plain 2D buffer this is 0; for a versioned 3D buffer this
  // resolves to stage_idx * pixel * channel — no special-casing needed.
  PrimExpr flat_offset = IntImm(DataType::Int(32), 0);
  {
    PrimExpr stride = IntImm(DataType::Int(32), 1);
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
      flat_offset = flat_offset + dstRegion_->region[i]->min * stride;
      stride = stride * dst_->shape[i];
    }
  }
  PrimExpr tile_elems =
      IntImm(DataType::Int(32), desc.smem_box_pixel * desc.smem_box_channel);
  PrimExpr shared_addr = dst_buffer.access_ptr(
      /*access_mask=*/2, /*dtype=*/DataType::Handle(), /*content_lanes=*/1,
      /*offset=*/flat_offset, /*extent=*/tile_elems);
  args.push_back(shared_addr);
  for (auto coord : global_coords)
    args.push_back(coord);
  for (auto offset : image_offset)
    args.push_back(offset);
  args.push_back(this->eviction_policy_);
  args.push_back(2); // inner_cache_policy: cache_normal
  args.push_back(2); // outer_cache_policy: cache_normal
  Stmt tma_copy_stmt =
      Evaluate(Call(DataType::Handle(), tma_load_im2col(), args));

  if (barrier_base_id >= 0) {
    bool ws_barrier = annotations_.Get("barrier").has_value();
    // Total bytes transferred by im2col TMA copy
    PrimExpr total_bytes =
        IntImm(DataType::Int(32), desc.smem_box_pixel * desc.smem_box_channel *
                                      dst_->dtype.bytes());

    Stmt barrier_before_tma_stmt = Evaluate(Call(
        DataType::Handle(), mbarrier_expect_tx(), {mbar_handle, total_bytes}));

    if (ws_barrier) {
      // External barrier (WS pass or InjectSoftwarePipeline).
      // Build: expect_tx + tma_load [+ arrive if emit_arrive is set].
      Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy_stmt};
      if (auto emit_arrive_val = annotations_.Get("emit_arrive")) {
        if (Downcast<IntImm>(emit_arrive_val.value())->value != 0) {
          producer_seq.push_back(
              Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(),
                            {mbar_handle})));
        }
      }
      Stmt producer =
          IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                     SeqStmt(producer_seq));
      return producer;
    }

    Stmt barrier_after_tma_stmt = Evaluate(
        Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));

    // Thread-gated block: expect_tx + tma_load_im2col + arrive
    Stmt producer = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                               SeqStmt({barrier_before_tma_stmt, tma_copy_stmt,
                                        barrier_after_tma_stmt}));

    // Emit producer + wait pair for pipeline/WS passes.
    Stmt wait_stmt =
        Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                      {mbar_handle, GetCopyMbarPhaseExpr(annotations_, T)}));

    return SeqStmt({producer, wait_stmt});
  }

  Stmt tma_copy = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                             tma_copy_stmt);
  return tma_copy;
}

// Encodes the TMA im2col descriptor for create_tma_im2col_descriptor().
Array<PrimExpr> TMAIm2ColDesc::EncodeCallArgs() const {
  Array<PrimExpr> args;
  args.reserve(rank * 5 + 5);

  args.push_back(data_type);
  args.push_back(static_cast<int>(rank));
  args.push_back(global_addr);
  for (auto e : global_shape)
    args.push_back(e);
  for (auto e : global_stride)
    args.push_back(e);
  for (auto e : elem_stride)
    args.push_back(e);
  for (auto e : lower_corner)
    args.push_back(e);
  for (auto e : upper_corner)
    args.push_back(e);
  args.push_back(smem_box_pixel);
  args.push_back(smem_box_channel);
  args.push_back(interleave);
  args.push_back(swizzle);
  args.push_back(l2_promotion);
  args.push_back(oob_fill);

  return args;
}
