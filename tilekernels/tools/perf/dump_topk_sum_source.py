import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch_musa
import tile_kernels

scores = torch.randn((4001, 12, 6), dtype=torch.float32, device="musa")
tile_kernels.moe.topk_sum_and_topk_group_idx(scores, 2, 4)
