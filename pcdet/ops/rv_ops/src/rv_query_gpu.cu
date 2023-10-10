#include <math.h>
#include <stdio.h>
#include <ctime>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include "rv_query_gpu.h"
#include "cuda_utils.h"

__device__ void swap_float(float *x, float *y) {
    float tmp = *x;
    *x = *y;
    *y = tmp;
}

__device__ void swap_int(int *x, int *y) {
    int tmp = *x;
    *x = *y;
    *y = tmp;
}

__device__ void reheap(float *dist, int *idx, int k) {
    int root = 0;
    int child = root * 2 + 1;
    while (child < k) {
        if(child + 1 < k && dist[child + 1] > dist[child])
            child++;
        if(dist[root] > dist[child])
            return;
        swap_float(&dist[root], &dist[child]);
        swap_int(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}

__global__ void rv_knn_query_kernel(int batch_size, int pts_num, int feats_dim, int rv_h, int rv_w, int num_points_per_pixel,
                                    int nsample, float radius, int h_dilation, int w_dilation, int h_range, int w_range, 
                                    const float *xyz, const float *feats, const float *query_rv_xyz, const float *query_rv_feats, 
                                    const int *query_rv_coords, const int *rv_map, int *idx) {
    // params xyz: (N1 + N2 + ..., 3)
    // params query_rv_xyz: (M1 + M2 + ..., 3)
    // params query_rv_coords: (M1 + M2 + ..., 3)
    // params query_rv_feats: (M1 + M2 + ..., C)
    // params rv_map: (B, rv_h, rv_w, num_points_per_pixel)
    // output:
    //      idx: (M1 + M2, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= pts_num) 
        return;
    query_rv_coords += pt_idx * 3;
    query_rv_xyz += pt_idx * 3;
    query_rv_feats += pt_idx * feats_dim;
    idx += pt_idx * nsample;
    int bs_idx = query_rv_coords[0], row_idx = query_rv_coords[1], col_idx = query_rv_coords[2];
    if (bs_idx < 0 || bs_idx >= batch_size || row_idx < 0 || row_idx >= rv_h || col_idx < 0 || col_idx >= rv_w)
        return;
    float new_x = query_rv_xyz[0], new_y = query_rv_xyz[1], new_z = query_rv_xyz[2];

    float best_dist[64];
    int best_idx[64];
    for(int i = 0; i < nsample; i++) {
        best_dist[i] = 1e10;
        best_idx[i] = -1;
    }
    int cnt = 0;
    float radius2 = radius * radius;
    for (int dh = -h_range * h_dilation; dh <= h_range * h_dilation; dh += h_dilation) {
        int h_coord = row_idx + dh;
        if (h_coord < 0 || h_coord >= rv_h) continue;
        for (int dw = -w_range * w_dilation; dw <= w_range * w_dilation; dw += w_dilation) {
            int w_coord = col_idx + dw;
            if (w_coord < 0 || w_coord >= rv_w) continue;
            int rv_map_idx = bs_idx * rv_h * rv_w * num_points_per_pixel + h_coord * rv_w * num_points_per_pixel + w_coord * num_points_per_pixel;
            int assign_pts_num = rv_map[rv_map_idx];
            for (int i = 0; i < assign_pts_num && i < num_points_per_pixel - 1; ++i) {
                int neighbor_idx = rv_map[rv_map_idx + i + 1];
                if (neighbor_idx < 0) continue;
                float x_per = xyz[neighbor_idx * 3 + 0], y_per = xyz[neighbor_idx * 3 + 1], z_per = xyz[neighbor_idx * 3 + 2];
                float dist2 = (x_per - new_x) * (x_per - new_x) + (y_per - new_y) * (y_per - new_y) + (z_per - new_z) * (z_per - new_z);
                if (dist2 > radius2) continue;
                dist2 = 0;
                for (int j = 0; j < feats_dim; ++j) {
                    float f1 = query_rv_feats[j], f2 = feats[neighbor_idx * feats_dim + j];
                    dist2 += (f1 - f2) * (f1 - f2);
                }
                if (dist2 < best_dist[0]) {
                    best_dist[0] = dist2;
                    best_idx[0] = neighbor_idx;
                    reheap(best_dist, best_idx, nsample);
                    ++cnt;
                }
            }
        }
    }
    
    if (cnt == 0) idx[0] = -1;
    else {
        int cnt2 = 0;
        for (int i = 0; i < nsample; ++i) {
            int b_idx = best_idx[i]; 
            if (b_idx != -1) idx[cnt2++] = b_idx;
        }
        for (int i = cnt2; i < nsample; ++i)
            idx[i] = idx[i % cnt2];
    }
}

__global__ void ball_rand_query_kernel(int B, int M, float radius, int nsample, \
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

    new_xyz += pt_idx * 3;
    xyz += xyz_batch_start_idx * 3;
    idx += pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0], new_y = new_xyz[1], new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0], y = xyz[k * 3 + 1], z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k + xyz_batch_start_idx;
                }
            }
            idx[cnt] = k + xyz_batch_start_idx;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    if (cnt == 0) idx[0] = -1;
}

__global__ void rv_query_kernel(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_map,
                                int *idx, int *sampled_pts_num) {
    // params xyz: (N1 + N2 + ..., 3)
    // params query_rv_xyz: (M1 + M2 + ..., 3)
    // params query_rv_coords: (M1 + M2 + ..., 3)
    // params rv_map: (B, rv_h, rv_w, num_points_per_pixel)
    // output:
    //      idx: (M1 + M2, nsample)
    //      sampled_pts_num: (M1 + M2, )
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= pts_num) 
        return;
    query_rv_coords += pt_idx * 3;
    query_rv_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    sampled_pts_num += pt_idx;
    int bs_idx = query_rv_coords[0], row_idx = query_rv_coords[1], col_idx = query_rv_coords[2];
    if (bs_idx < 0 || bs_idx >= batch_size || row_idx < 0 || row_idx >= rv_h || col_idx < 0 || col_idx >= rv_w)
        return;
    float new_x = query_rv_xyz[0], new_y = query_rv_xyz[1], new_z = query_rv_xyz[2];
    float radius2 = radius * radius;
    int cnt = 0;
    for (int dh = -h_range; dh <= h_range && cnt < nsample; dh += 1) {
        int h_coord = row_idx + dh * h_dilation;
        if (h_coord < 0 || h_coord >= rv_h) continue;
        for (int dw = -w_range; dw <= w_range && cnt < nsample; dw += 1) {
            int w_coord = col_idx + dw * w_dilation;
            if (w_coord < 0 || w_coord >= rv_w) continue;
            int rv_map_idx = bs_idx * rv_h * rv_w * num_points_per_pixel + h_coord * rv_w * num_points_per_pixel + w_coord * num_points_per_pixel;
            int assign_pts_num = rv_map[rv_map_idx];
            for (int i = 0; i < assign_pts_num && i < num_points_per_pixel - 1 && cnt < nsample; ++i) {
                int neighbor_idx = rv_map[rv_map_idx + i + 1];
                if (neighbor_idx < 0) continue;
                float x_per = xyz[neighbor_idx * 3 + 0], y_per = xyz[neighbor_idx * 3 + 1], z_per = xyz[neighbor_idx * 3 + 2];
                float dist2 = (x_per - new_x) * (x_per - new_x) + (y_per - new_y) * (y_per - new_y) + (z_per - new_z) * (z_per - new_z);
                if (dist2 > radius2) continue;
                if (cnt == 0) {
                    for (int l = 0; l < nsample; ++l) {
                        idx[l] = neighbor_idx;
                    }
                }
                idx[cnt] = neighbor_idx;
                ++cnt;
            }
        }
    }
    sampled_pts_num[0] = cnt;
}

__device__ void __update(float *dists, int *dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void rv_fps_kernel(int pts_num, int sampled_pts_num, int fps_pts_num,
                              const float *xyz, const int *pts_assign, const int *pooled_pts_num,
                              float *temp, int *pooled_pts_idx) {
    // params xyz: (N1 + N2 + ..., 3)
    // params temp: (M1 + M2 + ..., 8192)
    // params pts_assign: (M1 + M2 + ..., 8192)
    // params pooled_pts_num: (M1 + M2 + ...)
    // params pooled_pts_idx: (M1 + M2 + ..., 512)
    int stack_box_idx = blockIdx.y * gridDim.x + blockIdx.x;
    if (stack_box_idx >= pts_num) 
        return;
    int valid_sampled_pts_num = pooled_pts_num[stack_box_idx];
    if (valid_sampled_pts_num == 0) return;

    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = pts_assign[stack_box_idx * sampled_pts_num + 0];
    if (tid == 0)
        pooled_pts_idx[stack_box_idx * fps_pts_num + 0] = old;
    
    int j = 1;
    for (; j < fps_pts_num && j < valid_sampled_pts_num; j++) {
        int besti = 0;
        float best = -1;
        float x1 = xyz[old * 3 + 0];
        float y1 = xyz[old * 3 + 1];
        float z1 = xyz[old * 3 + 2];
        for (int k = tid; k < sampled_pts_num && k < valid_sampled_pts_num; k += stride) {
            int assign_idx = stack_box_idx * sampled_pts_num + k;
            int k_ = pts_assign[assign_idx];
            float x2 = xyz[k_ * 3 + 0];
            float y2 = xyz[k_ * 3 + 1];
            float z2 = xyz[k_ * 3 + 2];
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = temp[assign_idx];  // NO READ-WRITE CONFLICT
            if (j == 1 || d < d2) {
                temp[assign_idx] = d;
                d2 = d;
            }
            besti = d2 > best ? k_ : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 1024) {
            if (tid < 512) {
                __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();
        }
        if (block_size >= 512) {
            if (tid < 256) {
                __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
                __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
                __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
                __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
                __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
                __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
                __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
                __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
                __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            pooled_pts_idx[stack_box_idx * fps_pts_num + j] = old;
    }

    if (tid == 0) {
        for (int i = j; i < fps_pts_num; i++) {
            pooled_pts_idx[stack_box_idx * fps_pts_num + i] = pooled_pts_idx[stack_box_idx * fps_pts_num + i % j];
        }
    }
}

__global__ void rv_conv_query_kernel(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_map, int *idx) {
    // params xyz: (N1 + N2 + ..., 3)
    // params query_rv_xyz: (M1 + M2 + ..., 3)
    // params query_rv_coords: (M1 + M2 + ..., 3)
    // params rv_map: (B, rv_h, rv_w, num_points_per_pixel)
    // output:
    //      idx: (M1 + M2, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= pts_num)
        return;
    query_rv_coords += pt_idx * 3;
    query_rv_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    int bs_idx = query_rv_coords[0], row_idx = query_rv_coords[1], col_idx = query_rv_coords[2];
    if (bs_idx < 0 || bs_idx >= batch_size || row_idx < 0 || row_idx >= rv_h || col_idx < 0 || col_idx >= rv_w)
        return;
    float new_x = query_rv_xyz[0], new_y = query_rv_xyz[1], new_z = query_rv_xyz[2];
    float radius2 = radius * radius;
    for (int dh = -h_range; dh <= h_range; dh += 1) {
        int h_coord = row_idx + dh * h_dilation;
        if (h_coord < 0 || h_coord >= rv_h) continue;
        for (int dw = -w_range; dw <= w_range; dw += 1) {
            int w_coord = col_idx + dw * w_dilation;
            if (w_coord < 0 || w_coord >= rv_w) continue;
            int rv_map_idx = bs_idx * rv_h * rv_w * num_points_per_pixel + h_coord * rv_w * num_points_per_pixel + w_coord * num_points_per_pixel;
            int assign_pts_num = rv_map[rv_map_idx];
            for (int i = 0; i < assign_pts_num && i < num_points_per_pixel - 1; ++i) {
                int neighbor_idx = rv_map[rv_map_idx + i + 1];
                if (neighbor_idx < 0) continue;
                float x_per = xyz[neighbor_idx * 3 + 0], y_per = xyz[neighbor_idx * 3 + 1], z_per = xyz[neighbor_idx * 3 + 2];
                float dist2 = (x_per - new_x) * (x_per - new_x) + (y_per - new_y) * (y_per - new_y) + (z_per - new_z) * (z_per - new_z);
                if (dist2 > radius2) continue;
                idx[(dh + h_range) * (2 * w_range + 1) + dw + w_range] = neighbor_idx;
            }
        }
    }
}

__device__ float generate(curandState *globalState, int ind) {
	curandState localState = globalState[ind];
	float randf = curand_uniform(&localState);  // uniform distribution
	globalState[ind] = localState;
	return randf;
}

__global__ void setup_kernel(curandState *state, int pts_num, unsigned long seed)
{
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= pts_num) 
        return;
    curand_init(seed, pt_idx, 0, &state[pt_idx]);
}

__global__ void rv_rand_query_kernel(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                     float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                     const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_map, 
                                     int *idx, int *sampled_pts_num, curandState *globalState) {
    // params xyz: (N1 + N2 + ..., 3)
    // params query_rv_xyz: (M1 + M2 + ..., 3)
    // params query_rv_coords: (M1 + M2 + ..., 3)
    // params rv_map: (B, rv_h, rv_w, num_points_per_pixel)
    // output:
    //      idx: (M1 + M2, nsample)
    //      sampled_pts_num: (M1 + M2, )
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= pts_num) 
        return;
    query_rv_coords += pt_idx * 3;
    query_rv_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    sampled_pts_num += pt_idx;
    int bs_idx = query_rv_coords[0], row_idx = query_rv_coords[1], col_idx = query_rv_coords[2];
    if (bs_idx < 0 || bs_idx >= batch_size || row_idx < 0 || row_idx >= rv_h || col_idx < 0 || col_idx >= rv_w)
        return;
    float new_x = query_rv_xyz[0], new_y = query_rv_xyz[1], new_z = query_rv_xyz[2];
    float radius2 = radius * radius;
    int cnt = 0;
    for (int dh = -h_range; dh <= h_range; dh += 1) {
        int h_coord = row_idx + dh * h_dilation;
        if (h_coord < 0 || h_coord >= rv_h) continue;
        for (int dw = -w_range; dw <= w_range; dw += 1) {
            int w_coord = col_idx + dw * w_dilation;
            if (w_coord < 0 || w_coord >= rv_w) continue;
            int rv_map_idx = bs_idx * rv_h * rv_w * num_points_per_pixel + h_coord * rv_w * num_points_per_pixel + w_coord * num_points_per_pixel;
            int assign_pts_num = rv_map[rv_map_idx];
            for (int i = 0; i < assign_pts_num && i < num_points_per_pixel - 1; ++i) {
                int neighbor_idx = rv_map[rv_map_idx + i + 1];
                if (neighbor_idx < 0) continue;
                float x_per = xyz[neighbor_idx * 3 + 0], y_per = xyz[neighbor_idx * 3 + 1], z_per = xyz[neighbor_idx * 3 + 2];
                float dist2 = (x_per - new_x) * (x_per - new_x) + (y_per - new_y) * (y_per - new_y) + (z_per - new_z) * (z_per - new_z);
                if (dist2 > radius2) continue;
                if (cnt < nsample) {
                    if (cnt == 0) {
                        for (int l = 0; l < nsample; ++l) {
                            idx[l] = neighbor_idx;
                        }
                    }
                    idx[cnt] = neighbor_idx;
                } else {
                    int randi = (int)truncf(generate(globalState, pt_idx) * (cnt + 0.999999));  // [0, cnt]
                    if (randi < nsample) {
                        idx[randi] = neighbor_idx;
                    }
                }
                ++cnt;
            }
        }
    }
    sampled_pts_num[0] = cnt;
}

__global__ void rv_rand_query_kernel_v2(int batch_size, int pts_num, int rv_length, int rv_h, int rv_w, float radius, int nsample, 
                                        int h_dilation, int w_dilation, int h_range, int w_range, 
                                        const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_ends, 
                                        int *idx, int *sampled_pts_num, curandState *globalState) {
    // params xyz: (N1 + N2 + ..., 3)
    // params query_rv_xyz: (M1 + M2 + ..., 3)
    // params query_rv_coords: (M1 + M2 + ..., 3)
    // params rv_ends: (N,)
    // output:
    //      idx: (M1 + M2, nsample)
    //      sampled_pts_num: (M1 + M2, )
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= pts_num) 
        return;
    query_rv_coords += pt_idx * 3;
    query_rv_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    sampled_pts_num += pt_idx;
    int bs_idx = query_rv_coords[0], row_idx = query_rv_coords[1], col_idx = query_rv_coords[2];
    if (bs_idx < 0 || bs_idx >= batch_size || row_idx < 0 || row_idx >= rv_h || col_idx < 0 || col_idx >= rv_w)
        return;
    float new_x = query_rv_xyz[0], new_y = query_rv_xyz[1], new_z = query_rv_xyz[2];
    float radius2 = radius * radius;
    int cnt = 0;
    for (int dh = -h_range; dh <= h_range; dh += 1) {
        int h_coord = row_idx + dh * h_dilation;
        if (h_coord < 0 || h_coord >= rv_h) continue;
        for (int dw = -w_range; dw <= w_range; dw += 1) {
            int w_coord = col_idx + dw * w_dilation;
            if (w_coord < 0 || w_coord >= rv_w) continue;
            int rv_end_idx = bs_idx * rv_h * rv_w + h_coord * rv_w + w_coord;
            if (rv_end_idx >= rv_length) continue;
            int start_ = rv_end_idx == 0 ? 0 : rv_ends[rv_end_idx - 1];
            int end_ = rv_ends[rv_end_idx];
            for (int i = start_; i < end_; ++i) {
                float x_per = xyz[i * 3 + 0], y_per = xyz[i * 3 + 1], z_per = xyz[i * 3 + 2];
                float dist2 = (x_per - new_x) * (x_per - new_x) + (y_per - new_y) * (y_per - new_y) + (z_per - new_z) * (z_per - new_z);
                if (dist2 > radius2) continue;
                if (cnt < nsample) {
                    if (cnt == 0) {
                        for (int l = 0; l < nsample; ++l) {
                            idx[l] = i;
                        }
                    }
                    idx[cnt] = i;
                } else {
                    int randi = (int)truncf(generate(globalState, pt_idx) * (cnt + 0.999999));  // [0, cnt]
                    if (randi < nsample) {
                        idx[randi] = i;
                    }
                }
                ++cnt;
            }
        }
    }
    sampled_pts_num[0] = cnt;
}

void rv_knn_query_kernel_launcher(int batch_size, int pts_num, int feats_dim, int rv_h, int rv_w, int num_points_per_pixel,
                                  int nsample, float radius, int h_dilation, int w_dilation, int h_range, int w_range, 
                                  const float *xyz, const float *feats, const float *query_rv_xyz, const float *query_rv_feats, 
                                  const int *query_rv_coords, const int *rv_map, int *idx) {
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    rv_knn_query_kernel<<<blocks, threads>>>(batch_size, pts_num, feats_dim, rv_h, rv_w, num_points_per_pixel, 
                                             nsample, radius, h_dilation, w_dilation, h_range, w_range, 
                                             xyz, feats, query_rv_xyz, query_rv_feats, query_rv_coords, rv_map, idx);
}

void ball_rand_query_kernel_launcher(int B, int M, float radius, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_rand_query_kernel<<<blocks, threads>>>(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
}

void rv_fps_query_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                  float radius, int max_nsample, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                  const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_map,
                                  int *idx, int *sampled_pts_num) {
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    int *idx_temp = NULL;
    cudaMalloc(&idx_temp, pts_num * max_nsample * sizeof(int));
    // rv_query_kernel<<<blocks, threads>>>(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, max_nsample,
    //     h_dilation, w_dilation, h_range, w_range, xyz, query_rv_xyz, query_rv_coords, rv_map, idx_temp, sampled_pts_num);
    curandState* devStates;
	cudaMalloc(&devStates, pts_num * sizeof(curandState));
	setup_kernel<<<blocks, threads>>>(devStates, pts_num, unsigned(time(NULL)));
    rv_rand_query_kernel<<<blocks, threads>>>(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, max_nsample,
        h_dilation, w_dilation, h_range, w_range, xyz, query_rv_xyz, query_rv_coords, rv_map, idx_temp, sampled_pts_num, devStates);

    float *temp = NULL;
    cudaMalloc(&temp, pts_num * max_nsample * sizeof(float));

    dim3 blocks2(DIVUP(pts_num, batch_size), batch_size);
    rv_fps_kernel<THREADS_PER_BLOCK><<<blocks2, threads>>>(pts_num, max_nsample, nsample, xyz, idx_temp, sampled_pts_num, temp, idx);

    cudaFree(idx_temp);
    cudaFree(temp);
    cudaFree(devStates);
}

void rv_fps_query_kernel_launcher_v2(int batch_size, int pts_num, int rv_length, int rv_h, int rv_w, 
                                     float radius, int max_nsample, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                     const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_ends,
                                     int *idx, int *sampled_pts_num) {
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    int *idx_temp = NULL;
    cudaMalloc(&idx_temp, pts_num * max_nsample * sizeof(int));
    curandState* devStates;
	cudaMalloc(&devStates, pts_num * sizeof(curandState));
	setup_kernel<<<blocks, threads>>>(devStates, pts_num, unsigned(time(NULL)));
    rv_rand_query_kernel_v2<<<blocks, threads>>>(batch_size, pts_num, rv_length, rv_h, rv_w, radius, max_nsample,
        h_dilation, w_dilation, h_range, w_range, xyz, query_rv_xyz, query_rv_coords, rv_ends, idx_temp, sampled_pts_num, devStates);

    float *temp = NULL;
    cudaMalloc(&temp, pts_num * max_nsample * sizeof(float));

    dim3 blocks2(DIVUP(pts_num, batch_size), batch_size);
    rv_fps_kernel<THREADS_PER_BLOCK><<<blocks2, threads>>>(pts_num, max_nsample, nsample, xyz, idx_temp, sampled_pts_num, temp, idx);

    cudaFree(idx_temp);
    cudaFree(temp);
    cudaFree(devStates);
}

void rv_query_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                              float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                              const float *xyz, const float *query_rv_xyz, const int *query_rv_coords,
                              const int *rv_map, int *idx, int *sampled_pts_num) {
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    rv_query_kernel<<<blocks, threads>>>(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, nsample,
        h_dilation, w_dilation, h_range, w_range, xyz, query_rv_xyz, query_rv_coords, rv_map, idx, sampled_pts_num);
}

void rv_conv_query_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                   float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                   const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_map, int *idx) {
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    rv_conv_query_kernel<<<blocks, threads>>>(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, nsample,
                                              h_dilation, w_dilation, h_range, w_range, xyz, query_rv_xyz, query_rv_coords, rv_map, idx);
}

void rv_rand_query_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                   float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                   const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, 
                                   const int *rv_map, int *idx, int *sampled_pts_num) {
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    curandState* devStates;
	cudaMalloc(&devStates, pts_num * sizeof(curandState));
	setup_kernel<<<blocks, threads>>>(devStates, pts_num, unsigned(time(NULL)));
    rv_rand_query_kernel<<<blocks, threads>>>(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, nsample,
        h_dilation, w_dilation, h_range, w_range, xyz, query_rv_xyz, query_rv_coords, rv_map, idx, sampled_pts_num, devStates);
    cudaFree(devStates);
}
