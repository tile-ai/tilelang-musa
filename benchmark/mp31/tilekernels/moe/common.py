from tilelang import language as T


@T.macro
def get_topk_group_idx(
    scores_shared: T.SharedBuffer,
    topk_group_idx_shared: T.SharedBuffer,
    num_groups: int,
    num_experts_per_group: int,
    num_topk_groups: int,
    num_topk_sum: int,
    num_vectorize_for_grouped_expert: int,
):
    thread_idx = T.get_thread_binding()
    token_idx = thread_idx // 32
    lane_idx = thread_idx % 32
    scores_vec_local = T.alloc_local((num_vectorize_for_grouped_expert,), dtype=T.float32)

    top1_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    top2_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    topk_sum_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    count_var = T.alloc_var(dtype=T.int32, init=0)

    # Get the topk sum of each group
    if lane_idx < num_groups:
        num_vec_experts_per_group = num_experts_per_group // num_vectorize_for_grouped_expert
        for i in T.unroll(num_vec_experts_per_group):
            for j in T.vectorized(num_vectorize_for_grouped_expert):
                # Shift to avoid bank conflict
                vec_idx = (i + lane_idx) % num_vec_experts_per_group
                scores_vec_local[j] = scores_shared[
                    token_idx, lane_idx * num_experts_per_group + vec_idx * num_vectorize_for_grouped_expert + j
                ]
                if scores_vec_local[j] > top1_var:
                    top2_var = top1_var
                    top1_var = scores_vec_local[j]
                elif scores_vec_local[j] > top2_var:
                    top2_var = scores_vec_local[j]
        topk_sum_var = T.Select(num_topk_sum == 1, top1_var, top1_var + top2_var)

    # Count the number of groups that have a larger top2 sum
    for i in T.unroll(num_groups):
        other_top2_sum = T.shfl_sync(0xFFFFFFFF, topk_sum_var, i)
        if other_top2_sum > topk_sum_var or (other_top2_sum == topk_sum_var and i < lane_idx):
            count_var += 1

    # Get the topk groups in group_idx order for stable sort
    if count_var < num_topk_groups:
        topk_group_idx_shared[token_idx, count_var] = lane_idx

    # Sync warp to ensure all threads have written their topk group indices
    T.sync_warp()


@T.macro
def write_topk_group_idx_global(
    scores: T.Buffer,
    group_topk_idx: T.Buffer,
    global_token_idx: T.int32,
    num_groups: int,
    num_experts_per_group: int,
    num_topk_groups: int,
    num_topk_sum: int,
    num_vectorize_for_grouped_expert: int,
):
    lane_idx = T.get_thread_binding() % 32
    global_token_idx_i64 = T.alloc_var(dtype=T.int64, init=T.cast(global_token_idx, T.int64))
    scores_vec_local = T.alloc_local((num_vectorize_for_grouped_expert,), dtype=T.float32)

    top1_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    top2_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    topk_sum_var = T.alloc_var(dtype=T.float32, init=-T.infinity(T.float32))
    count_var = T.alloc_var(dtype=T.int32, init=0)

    if lane_idx < num_groups:
        num_vec_experts_per_group = num_experts_per_group // num_vectorize_for_grouped_expert
        for i in T.unroll(num_vec_experts_per_group):
            for j in T.vectorized(num_vectorize_for_grouped_expert):
                vec_idx = i
                scores_vec_local[j] = scores[
                    global_token_idx_i64,
                    lane_idx * num_experts_per_group + vec_idx * num_vectorize_for_grouped_expert + j,
                ]
                if scores_vec_local[j] > top1_var:
                    top2_var = top1_var
                    top1_var = scores_vec_local[j]
                elif scores_vec_local[j] > top2_var:
                    top2_var = scores_vec_local[j]
        topk_sum_var = T.Select(num_topk_sum == 1, top1_var, top1_var + top2_var)

    for i in T.unroll(num_groups):
        other_top2_sum = T.shfl_sync(0xFFFFFFFF, topk_sum_var, i)
        if other_top2_sum > topk_sum_var or (other_top2_sum == topk_sum_var and i < lane_idx):
            count_var += 1

    if count_var < num_topk_groups:
        group_topk_idx[global_token_idx_i64, count_var] = lane_idx
