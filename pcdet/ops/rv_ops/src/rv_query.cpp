#include <torch/serialize/tensor.h>
#include <vector>
// #include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "rv_query_gpu.h"

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int rv_knn_query_wrapper(int batch_size, int pts_num, int feats_dim, int rv_h, int rv_w, int num_points_per_pixel,
                         int nsample, float radius, int h_dilation, int w_dilation, int h_range, int w_range,
                         at::Tensor xyz_tensor, at::Tensor feats_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_feats_tensor, 
                         at::Tensor query_rv_coords_tensor, at::Tensor rv_map_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(feats_tensor);
    CHECK_INPUT(query_rv_xyz_tensor);
    CHECK_INPUT(query_rv_feats_tensor);
    CHECK_INPUT(query_rv_coords_tensor);
    CHECK_INPUT(rv_map_tensor);
    CHECK_INPUT(idx_tensor);

    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *feats = feats_tensor.data_ptr<float>();
    const float *query_rv_xyz = query_rv_xyz_tensor.data_ptr<float>();
    const float *query_rv_feats = query_rv_feats_tensor.data_ptr<float>();
    const int *query_rv_coords = query_rv_coords_tensor.data_ptr<int>();
    const int *rv_map = rv_map_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();

    rv_knn_query_kernel_launcher(batch_size, pts_num, feats_dim, rv_h, rv_w, num_points_per_pixel, nsample, 
                                 radius, h_dilation, w_dilation, h_range, w_range, 
                                 xyz, feats, query_rv_xyz, query_rv_feats, query_rv_coords, rv_map, idx);
    return 1;
}

int ball_rand_query_wrapper(int batch_size, int pts_num, float radius, int nsample, 
                     at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor, at::Tensor xyz_tensor,
                     at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(new_xyz_batch_cnt_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(idx_tensor);

    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data_ptr<int>();
    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();

    ball_rand_query_kernel_launcher(batch_size, pts_num, radius, nsample, new_xyz, 
                new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
    return 1;
}

int rv_fps_query_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                         float radius, int max_nsample, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                         at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                         at::Tensor rv_map_tensor, at::Tensor idx_tensor, at::Tensor sampled_pts_num_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(query_rv_xyz_tensor);
    CHECK_INPUT(query_rv_coords_tensor);
    CHECK_INPUT(rv_map_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(sampled_pts_num_tensor);

    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *query_rv_xyz = query_rv_xyz_tensor.data_ptr<float>();
    const int *query_rv_coords = query_rv_coords_tensor.data_ptr<int>();
    const int *rv_map = rv_map_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    int *sampled_pts_num = sampled_pts_num_tensor.data_ptr<int>();

    rv_fps_query_kernel_launcher(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, 
                                 max_nsample, nsample, h_dilation, w_dilation, h_range, w_range, 
                                 xyz, query_rv_xyz, query_rv_coords, rv_map, idx, sampled_pts_num);
    return 1;
}

int rv_fps_query_wrapper_v2(int batch_size, int pts_num, int rv_length, int rv_h, int rv_w,
                            float radius, int max_nsample, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                            at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                            at::Tensor rv_ends_tensor, at::Tensor idx_tensor, at::Tensor sampled_pts_num_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(query_rv_xyz_tensor);
    CHECK_INPUT(query_rv_coords_tensor);
    CHECK_INPUT(rv_ends_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(sampled_pts_num_tensor);

    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *query_rv_xyz = query_rv_xyz_tensor.data_ptr<float>();
    const int *query_rv_coords = query_rv_coords_tensor.data_ptr<int>();
    const int *rv_ends = rv_ends_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    int *sampled_pts_num = sampled_pts_num_tensor.data_ptr<int>();

    rv_fps_query_kernel_launcher_v2(batch_size, pts_num, rv_length, rv_h, rv_w, radius, 
                                    max_nsample, nsample, h_dilation, w_dilation, h_range, w_range, 
                                    xyz, query_rv_xyz, query_rv_coords, rv_ends, idx, sampled_pts_num);
    return 1;
}

int rv_query_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                     float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                     at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                     at::Tensor rv_map_tensor, at::Tensor idx_tensor, at::Tensor sampled_pts_num_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(query_rv_xyz_tensor);
    CHECK_INPUT(query_rv_coords_tensor);
    CHECK_INPUT(rv_map_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(sampled_pts_num_tensor);

    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *query_rv_xyz = query_rv_xyz_tensor.data_ptr<float>();
    const int *query_rv_coords = query_rv_coords_tensor.data_ptr<int>();
    const int *rv_map = rv_map_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    int *sampled_pts_num = sampled_pts_num_tensor.data_ptr<int>();

    rv_query_kernel_launcher(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, nsample, 
                             h_dilation, w_dilation, h_range, w_range, 
                             xyz, query_rv_xyz, query_rv_coords, rv_map, idx, sampled_pts_num);
    return 1;
}

int rv_conv_query_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                     float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                     at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                     at::Tensor rv_map_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(query_rv_xyz_tensor);
    CHECK_INPUT(query_rv_coords_tensor);
    CHECK_INPUT(rv_map_tensor);
    CHECK_INPUT(idx_tensor);

    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *query_rv_xyz = query_rv_xyz_tensor.data_ptr<float>();
    const int *query_rv_coords = query_rv_coords_tensor.data_ptr<int>();
    const int *rv_map = rv_map_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();

    rv_conv_query_kernel_launcher(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, nsample, 
                             h_dilation, w_dilation, h_range, w_range, 
                             xyz, query_rv_xyz, query_rv_coords, rv_map, idx);
    return 1;
}

int rv_rand_query_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                          float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                          at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                          at::Tensor rv_map_tensor, at::Tensor idx_tensor, at::Tensor sampled_pts_num_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(query_rv_xyz_tensor);
    CHECK_INPUT(query_rv_coords_tensor);
    CHECK_INPUT(rv_map_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(sampled_pts_num_tensor);

    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *query_rv_xyz = query_rv_xyz_tensor.data_ptr<float>();
    const int *query_rv_coords = query_rv_coords_tensor.data_ptr<int>();
    const int *rv_map = rv_map_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    int *sampled_pts_num = sampled_pts_num_tensor.data_ptr<int>();

    rv_rand_query_kernel_launcher(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, radius, nsample, 
                                  h_dilation, w_dilation, h_range, w_range, 
                                  xyz, query_rv_xyz, query_rv_coords, rv_map, idx, sampled_pts_num);
    return 1;
}
