/*
Center assignments
Written by Jiageng Mao
*/

#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

__device__ __forceinline__ static void reduceMax(float *address, float val) {
    int *address_as_i = reinterpret_cast<int *>(address);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old || __int_as_float(old) < val);
}

__device__ float gaussian_radius(float height, float width, float min_overlap){
    float a1 = 1;
    float b1 = (height + width);
    float c1 = width * height * (1 - min_overlap) / (1 + min_overlap);
    float sq1 = sqrt(b1 * b1 - 4 * a1 * c1);
    float r1 = (b1 + sq1) / 2;

    float a2 = 4;
    float b2 = 2 * (height + width);
    float c2 = (1 - min_overlap) * width * height;
    float sq2 = sqrt(b2 * b2 - 4 * a2 * c2);
    float r2 = (b2 + sq2) / 2;

    float a3 = 4 * min_overlap;
    float b3 = -2 * min_overlap * (height + width);
    float c3 = (min_overlap - 1) * width * height;
    float sq3 = sqrt(b3 * b3 - 4 * a3 * c3);
    float r3 = (b3 + sq3) / 2;
    return min(min(r1, r2), r3);
}

__global__ void draw_center_kernel(int batch_size, int max_boxes, int max_objs, int num_cls, int H, int W, int min_radius, float out_factor, float gaussian_overlap, 
                                   const float *gt_boxes, float *heatmap, int *gt_ind, int *gt_mask, int *gt_cat, float *gt_box_encoding, int *gt_cnt){

    /*
        Args:
            gt_boxes: (B, max_boxes, 5) with class labels
            heatmap: (B, num_cls, H, W)
            gt_ind: (B, num_cls, max_objs)
            gt_mask: (B, num_cls, max_objs)
            gt_cat: (B, num_cls, max_objs)
            gt_box_encoding: (B, num_cls, max_objs, 4)
            gt_cnt: (B, num_cls)
    */
    int bs_idx = blockIdx.y;
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || box_idx >= max_boxes) return;

    // move pointer
    gt_boxes += bs_idx * max_boxes * 5;
    heatmap += bs_idx * num_cls * H * W;
    gt_ind += bs_idx * num_cls * max_objs;
    gt_mask += bs_idx * num_cls * max_objs;
    gt_cat += bs_idx * num_cls * max_objs;
    gt_box_encoding += bs_idx * num_cls * max_objs * 4;
    gt_cnt += bs_idx * num_cls;

    // gt box parameters
    float x1 = gt_boxes[box_idx * 5 + 0];
    float y1 = gt_boxes[box_idx * 5 + 1];
    float x2 = gt_boxes[box_idx * 5 + 2];
    float y2 = gt_boxes[box_idx * 5 + 3];
    int cls = gt_boxes[box_idx * 5 + 4];

    // box not defined
    if (cls == 0) return;

    // cls begin from 1
    int cls_idx = cls - 1;
    heatmap += cls_idx * H * W;
    gt_ind += cls_idx * max_objs;
    gt_mask += cls_idx * max_objs;
    gt_cat += cls_idx * max_objs;
    gt_box_encoding += cls_idx * max_objs * 4;
    gt_cnt += cls_idx;

    float w = (x2 - x1) / out_factor;
    float h = (y2 - y1) / out_factor;
    float radius = gaussian_radius(ceil(h), ceil(w), gaussian_overlap);
    int radius_int = max(min_radius, (int) radius);
    float c_x = (x1 + x2) / 2.0 / out_factor;
    float c_y = (y1 + y2) / 2.0 / out_factor;
    int c_x_int = (int) c_x;
    int c_y_int = (int) c_y;
    if (c_x_int >= W || c_x_int < 0) return;
    if (c_y_int >= H || c_y_int < 0) return;

    // draw gaussian map
    float div_factor = 6.0;
    float sigma = (2 * radius_int + 1) / div_factor;
    for (int scan_y = -radius_int; scan_y < radius_int + 1; scan_y++){
        if (c_y_int + scan_y < 0 || c_y_int + scan_y >= H) continue;
        for (int scan_x = -radius_int; scan_x < radius_int + 1; scan_x++){
            if (c_x_int + scan_x < 0 || c_x_int + scan_x >= W) continue;
            float weight = exp(-(scan_x * scan_x + scan_y * scan_y) / (2 * sigma * sigma)); // force convert float sigma
            float eps = 0.0000001;
            if (weight < eps) weight = 0;
            float *w_addr = heatmap + (c_y_int + scan_y) * W + (c_x_int + scan_x);
            reduceMax(w_addr, weight);
        }
    }
    int obj_idx = atomicAdd(gt_cnt, 1);
    if (obj_idx >= max_objs) return;
    gt_ind[obj_idx] = c_y_int * W + c_x_int;
    gt_mask[obj_idx] = 1;
    gt_cat[obj_idx] = cls_idx + 1; // begin from 1
    gt_box_encoding[obj_idx * 4 + 0] = c_x - c_x_int;
    gt_box_encoding[obj_idx * 4 + 1] = c_y - c_y_int;
    gt_box_encoding[obj_idx * 4 + 2] = w;
    gt_box_encoding[obj_idx * 4 + 3] = h;
    return;
}

__global__ void draw_bev_all_kernel(int batch_size, int max_boxes, int num_cls, int H, int W, int code_size,
                                    int min_radius, float voxel_x, float voxel_y, float range_x, float range_y,
                                    float out_factor, float gaussian_overlap, const float *gt_boxes, const float *gt_corners, 
                                    float *center_map, float *corner_map) {

    /*
        Args:
            gt_boxes: (B, max_boxes, code_size) with class labels
            gt_corners: (B, max_boxes, 4, 2) 4 corner coords
            center_map: (B, num_cls, H, W)
            corner_map: (B, num_cls, 4, H, W) each corner has a seperate map
    */
    int bs_idx = blockIdx.y;
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || box_idx >= max_boxes) return;

    // move pointer
    gt_boxes += bs_idx * max_boxes * code_size;
    gt_corners += bs_idx * max_boxes * 4 * 2;
    center_map += bs_idx * num_cls * H * W;
    corner_map += bs_idx * num_cls * 4 * H * W;

    // gt box parameters
    float x = gt_boxes[box_idx * code_size + 0];
    float y = gt_boxes[box_idx * code_size + 1];
    float dx = gt_boxes[box_idx * code_size + 3];
    float dy = gt_boxes[box_idx * code_size + 4];
    int cls = gt_boxes[box_idx * code_size + code_size - 1];

    // box not defined
    if (cls == 0) return;

    // gt corner coordinates
    float corner_x0 = gt_corners[box_idx * 4 * 2 + 0], corner_y0 = gt_corners[box_idx * 4 * 2 + 1];
    float corner_x1 = gt_corners[box_idx * 4 * 2 + 2], corner_y1 = gt_corners[box_idx * 4 * 2 + 3];
    float corner_x2 = gt_corners[box_idx * 4 * 2 + 4], corner_y2 = gt_corners[box_idx * 4 * 2 + 5];
    float corner_x3 = gt_corners[box_idx * 4 * 2 + 6], corner_y3 = gt_corners[box_idx * 4 * 2 + 7];

    // cls begin from 1
    int cls_idx = cls - 1;
    center_map += cls_idx * H * W;
    corner_map += cls_idx * 4 * H * W;

    float coor_dx = dx / voxel_x / out_factor;
    float coor_dy = dy / voxel_y / out_factor;
    float radius = gaussian_radius(coor_dy, coor_dx, gaussian_overlap);
    // note that gaussian radius is shared both center_map and corner_map
    // this can be modified according to needs
    int radius_int = max(min_radius, (int) radius);
    int coor_x_int = (x - range_x) / voxel_x / out_factor;
    int coor_y_int = (y - range_y) / voxel_y / out_factor;
    // if center is outside, directly return
    if (coor_x_int >= W || coor_x_int < 0 || coor_y_int >= H || coor_y_int < 0) return;

    // transform corners to bev map coords
    int coor_corner_x0_int = (corner_x0 - range_x) / voxel_x / out_factor;
    int coor_corner_y0_int = (corner_y0 - range_y) / voxel_y / out_factor;
    int coor_corner_x1_int = (corner_x1 - range_x) / voxel_x / out_factor;
    int coor_corner_y1_int = (corner_y1 - range_y) / voxel_y / out_factor;
    int coor_corner_x2_int = (corner_x2 - range_x) / voxel_x / out_factor;
    int coor_corner_y2_int = (corner_y2 - range_y) / voxel_y / out_factor;
    int coor_corner_x3_int = (corner_x3 - range_x) / voxel_x / out_factor;
    int coor_corner_y3_int = (corner_y3 - range_y) / voxel_y / out_factor;

    // draw gaussian center_map and corner_map
    float div_factor = 6.0;
    float sigma = (2 * radius_int + 1) / div_factor;
    for (int scan_y = -radius_int; scan_y <= radius_int; scan_y++){
        for (int scan_x = -radius_int; scan_x <= radius_int; scan_x++){
            float weight = exp(-(scan_x * scan_x + scan_y * scan_y) / (2 * sigma * sigma)); // force convert float sigma
            float eps = 0.0000001;
            if (weight < eps) weight = 0;
            // draw center
            if (coor_x_int + scan_x >= 0 && coor_x_int + scan_x < W && coor_y_int + scan_y >= 0 && coor_y_int + scan_y < H) {
                float *center_addr = center_map + (coor_y_int + scan_y) * W + (coor_x_int + scan_x);
                reduceMax(center_addr, weight);
            }
            // draw 4 corners
            if (coor_corner_x0_int + scan_x >= 0 && coor_corner_x0_int + scan_x < W && coor_corner_y0_int + scan_y >= 0 && coor_corner_y0_int + scan_y < H) {
                float *corner_addr0 = corner_map + 0 * W * H + (coor_corner_y0_int + scan_y) * W + (coor_corner_x0_int + scan_x);
                reduceMax(corner_addr0, weight);
            }
            if (coor_corner_x1_int + scan_x >= 0 && coor_corner_x1_int + scan_x < W && coor_corner_y1_int + scan_y >= 0 && coor_corner_y1_int + scan_y < H) {
                float *corner_addr1 = corner_map + 1 * W * H + (coor_corner_y1_int + scan_y) * W + (coor_corner_x1_int + scan_x);
                reduceMax(corner_addr1, weight);
            }
            if (coor_corner_x2_int + scan_x >= 0 && coor_corner_x2_int + scan_x < W && coor_corner_y2_int + scan_y >= 0 && coor_corner_y2_int + scan_y < H) {
                float *corner_addr2 = corner_map + 2 * W * H + (coor_corner_y2_int + scan_y) * W + (coor_corner_x2_int + scan_x);
                reduceMax(corner_addr2, weight);
            }
            if (coor_corner_x3_int + scan_x >= 0 && coor_corner_x3_int + scan_x < W && coor_corner_y3_int + scan_y >= 0 && coor_corner_y3_int + scan_y < H) {
                float *corner_addr3 = corner_map + 3 * W * H + (coor_corner_y3_int + scan_y) * W + (coor_corner_x3_int + scan_x);
                reduceMax(corner_addr3, weight);
            }
        }
    }
    return;
}

void draw_center_kernel_launcher(int batch_size, int max_boxes, int max_objs, int num_cls, int H, int W, int min_radius, float out_factor, float gaussian_overlap,
                                 const float *gt_boxes, float *heatmap, int *gt_ind, int *gt_mask, int *gt_cat, float *gt_box_encoding, int *gt_cnt) {
    dim3 blocks(DIVUP(max_boxes, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    draw_center_kernel<<<blocks, threads>>>(batch_size, max_boxes, max_objs, num_cls, H, W, min_radius, out_factor, gaussian_overlap,
                                            gt_boxes, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt);
}

void draw_bev_all_kernel_launcher(int batch_size, int max_boxes, int num_cls, int H, int W, int code_size, int min_radius,
                                  float voxel_x, float voxel_y, float range_x, float range_y, float out_factor,
                                  float gaussian_overlap, const float *gt_boxes, const float *gt_corners,
                                  float *center_map, float *corner_map){
    dim3 blocks(DIVUP(max_boxes, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    draw_bev_all_kernel<<<blocks, threads>>>(batch_size, max_boxes, num_cls, H, W, code_size, min_radius,
                                             voxel_x, voxel_y, range_x, range_y, out_factor,
                                             gaussian_overlap, gt_boxes, gt_corners, center_map, corner_map);
}
