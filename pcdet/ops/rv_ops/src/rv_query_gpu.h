#ifndef _RV_QUERY_GPU_H
#define _RV_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int rv_knn_query_wrapper(int batch_size, int pts_num, int feats_dim, int rv_h, int rv_w, int num_points_per_pixel,
                         int nsample, float radius, int h_dilation, int w_dilation, int h_range, int w_range,
                         at::Tensor xyz_tensor, at::Tensor feats_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_feats_tensor, 
                         at::Tensor query_rv_coords_tensor, at::Tensor rv_map_tensor, at::Tensor idx_tensor);

void rv_knn_query_kernel_launcher(int batch_size, int pts_num, int feats_dim, int rv_h, int rv_w, int num_points_per_pixel,
                                  int nsample, float radius, int h_dilation, int w_dilation, int h_range, int w_range, 
                                  const float *xyz, const float *feats, const float *query_rv_xyz, 
                                  const float *query_rv_feats, const int *query_rv_coords, const int *rv_map, int *idx);

int ball_rand_query_wrapper(int B, int M, float radius, int nsample, 
                     at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor, at::Tensor xyz_tensor,
                     at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor);

void ball_rand_query_kernel_launcher(int B, int M, float radius, int nsample, const float *new_xyz, 
                const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx);

int rv_fps_query_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                         float radius, int max_nsample, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                         at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                         at::Tensor rv_map_tensor, at::Tensor idx_tensor, at::Tensor sampled_pts_num_tensor);

void rv_fps_query_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                  float radius, int max_nsample, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                  const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_map, int *idx, int *sampled_pts_num);

int rv_fps_query_wrapper_v2(int batch_size, int pts_num, int rv_length, int rv_h, int rv_w,
                            float radius, int max_nsample, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                            at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                            at::Tensor rv_ends_tensor, at::Tensor idx_tensor, at::Tensor sampled_pts_num_tensor);

void rv_fps_query_kernel_launcher_v2(int batch_size, int pts_num, int rv_length, int rv_h, int rv_w,
                                     float radius, int max_nsample, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                     const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_ends, int *idx, int *sampled_pts_num);

int rv_query_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                     float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                     at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                     at::Tensor rv_map_tensor, at::Tensor idx_tensor, at::Tensor sampled_pts_num_tensor);

void rv_query_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                              float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                              const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, 
                              const int *rv_map, int *idx, int *sampled_pts_num);

int rv_conv_query_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                          float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                          at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                          at::Tensor rv_map_tensor, at::Tensor idx_tensor);

void rv_conv_query_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                   float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                   const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, const int *rv_map, int *idx);

int rv_rand_query_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                          float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range,
                          at::Tensor xyz_tensor, at::Tensor query_rv_xyz_tensor, at::Tensor query_rv_coords_tensor,
                          at::Tensor rv_map_tensor, at::Tensor idx_tensor, at::Tensor sampled_pts_num_tensor);

void rv_rand_query_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel,
                                   float radius, int nsample, int h_dilation, int w_dilation, int h_range, int w_range, 
                                   const float *xyz, const float *query_rv_xyz, const int *query_rv_coords, 
                                   const int *rv_map, int *idx, int *sampled_pts_num);

#endif
