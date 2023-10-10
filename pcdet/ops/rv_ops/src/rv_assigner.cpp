#include <torch/serialize/tensor.h>
#include <vector>
// #include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "rv_assigner_gpu.h"

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

int rv_assigner_wrapper(int batch_size, int pts_num, int rv_h, int rv_w, int num_points_per_pixel, at::Tensor rv_coords_tensor, at::Tensor rv_map_tensor) {
    CHECK_INPUT(rv_coords_tensor);
    CHECK_INPUT(rv_map_tensor);
    
    const int *rv_coords = rv_coords_tensor.data_ptr<int>();
    int *rv_map = rv_map_tensor.data_ptr<int>();

    rv_assigner_kernel_launcher(batch_size, pts_num, rv_h, rv_w, num_points_per_pixel, rv_coords, rv_map);
    return 1;
}
