#pragma once

namespace tl {

__device__ inline unsigned int ceil_div(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

template <int panel_width> __device__ inline dim3 rasterization2DRow() {
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int grid_size = gridDim.x * gridDim.y;
  const unsigned int panel_size = panel_width * gridDim.x;
  const unsigned int panel_offset = block_idx % panel_size;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int total_panel = ceil_div(grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.x;
  const unsigned int col_idx = (panel_idx & 1)
                                   ? gridDim.x - 1 - panel_offset / stride
                                   : panel_offset / stride;
  const unsigned int row_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

template <int panel_width> __device__ inline dim3 rasterization2DColumn() {
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int grid_size = gridDim.x * gridDim.y;
  const unsigned int panel_size = panel_width * gridDim.y;
  const unsigned int panel_offset = block_idx % panel_size;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int total_panel = ceil_div(grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.y;
  const unsigned int row_idx = (panel_idx & 1)
                                   ? gridDim.y - 1 - panel_offset / stride
                                   : panel_offset / stride;
  const unsigned int col_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

// Generalized 2D Hilbert (Gilbert) space-filling curve mapping for arbitrary
// rectangular grids (gridDim.x x gridDim.y). Strictly bijective with unit
// Manhattan step between consecutive threadblocks to maximize L2 tile reuse.
template <int panel_width> __device__ inline dim3 rasterization2DHilbert() {
  int idx = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
  const int w = static_cast<int>(gridDim.x);
  const int h = static_cast<int>(gridDim.y);
  int x = 0;
  int y = 0;
  int ax = (w >= h) ? w : 0;
  int ay = (w >= h) ? 0 : h;
  int bx = (w >= h) ? 0 : w;
  int by = (w >= h) ? h : 0;

  while (true) {
    int s_a = ax + ay;
    int s_b = bx + by;
    int aw = s_a < 0 ? -s_a : s_a;
    int bh = s_b < 0 ? -s_b : s_b;
    int dax = (ax > 0) - (ax < 0);
    int day = (ay > 0) - (ay < 0);
    int dbx = (bx > 0) - (bx < 0);
    int dby = (by > 0) - (by < 0);

    if (bh == 1) {
      x += dax * idx;
      y += day * idx;
      break;
    }
    if (aw == 1) {
      x += dbx * idx;
      y += dby * idx;
      break;
    }

    int ax2 = ax / 2;
    int ay2 = ay / 2;
    int bx2 = bx / 2;
    int by2 = by / 2;
    int s_a2 = ax2 + ay2;
    int s_b2 = bx2 + by2;
    int w2 = s_a2 < 0 ? -s_a2 : s_a2;
    int h2 = s_b2 < 0 ? -s_b2 : s_b2;

    if (2 * aw > 3 * bh) {
      if ((w2 & 1) && (aw > 2)) {
        ax2 += dax;
        ay2 += day;
        int sa = ax2 + ay2;
        w2 = sa < 0 ? -sa : sa;
      }
      int cnt = w2 * bh;
      if (idx < cnt) {
        ax = ax2;
        ay = ay2;
      } else {
        idx -= cnt;
        x += ax2;
        y += ay2;
        ax -= ax2;
        ay -= ay2;
      }
    } else {
      if ((h2 & 1) && (bh > 2)) {
        bx2 += dbx;
        by2 += dby;
        int sb = bx2 + by2;
        h2 = sb < 0 ? -sb : sb;
      }
      int cnt1 = h2 * w2;
      if (idx < cnt1) {
        ax = bx2;
        ay = by2;
        bx = ax2;
        by = ay2;
      } else {
        idx -= cnt1;
        int cnt2 = aw * (bh - h2);
        if (idx < cnt2) {
          x += bx2;
          y += by2;
          bx -= bx2;
          by -= by2;
        } else {
          idx -= cnt2;
          int old_ax = ax;
          int old_ay = ay;
          x += (old_ax - dax) + (bx2 - dbx);
          y += (old_ay - day) + (by2 - dby);
          ax = -bx2;
          ay = -by2;
          bx = -(old_ax - ax2);
          by = -(old_ay - ay2);
        }
      }
    }
  }
  return {static_cast<unsigned int>(x), static_cast<unsigned int>(y),
          blockIdx.z};
}

} // namespace tl
