#include <math.h>
#include <stdio.h>

#include "rv_assigner_gpu.h"
#include "cuda_utils.h"


__global__ void rv_assigner_kernel(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel, const int *rv_coords, int *rv_map) {
    // params rv_coords: (N1 + N2 + ..., 3)
    // params rv_map: (B, rv_h, rv_w, num_points_per_pixel)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= pts_num) {
        return;
    }
    rv_coords += pt_idx * 3;
    int bs_idx = rv_coords[0], row_idx = rv_coords[1], col_idx = rv_coords[2];
    if (bs_idx < 0 || bs_idx >= batch_size || row_idx < 0 || row_idx >= rv_h || col_idx < 0 || col_idx >= rv_w) 
        return;
    rv_map += bs_idx * rv_h * rv_w * num_points_per_pixel + row_idx * rv_w * num_points_per_pixel + col_idx * num_points_per_pixel;
    int assign_pts_num = atomicAdd(rv_map, 1);
    if (assign_pts_num < num_points_per_pixel - 1)
        rv_map[assign_pts_num + 1] = pt_idx;
}

void rv_assigner_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel, const int *rv_coords, int *rv_map) {
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    rv_assigner_kernel<<<blocks, threads>>>(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, rv_coords, rv_map);
}
