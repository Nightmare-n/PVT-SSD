/*
Stacked-batch-data version of point grouping, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
// #include <THC/THC.h>
#include "rv_group_gpu.h"

// extern THCState *state;
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


int rv_group_grad_wrapper(int M, int C, int nsample,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_features_tensor) {

    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(grad_features_tensor);

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_features = grad_features_tensor.data<float>();

    rv_group_grad_kernel_launcher(M, C, nsample, grad_out, idx, grad_features);
    return 1;
}


int rv_group_wrapper(int M, int C, int nsample,
    at::Tensor features_tensor, at::Tensor idx_tensor, at::Tensor out_tensor) {

    CHECK_INPUT(features_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(out_tensor);

    const float *features = features_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();

    rv_group_kernel_launcher(M, C, nsample, features, idx, out);
    return 1;
}