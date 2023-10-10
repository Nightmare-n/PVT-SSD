#ifndef _RV_ASSIGNER_GPU_H
#define _RV_ASSIGNER_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int rv_assigner_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel, at::Tensor rv_coords_tensor, at::Tensor rv_map_tensor);

void rv_assigner_kernel_launcher(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel, const int *rv_coords, int *rv_map);

#endif
