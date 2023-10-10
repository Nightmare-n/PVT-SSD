/*
Stacked-batch-data version of point grouping, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#ifndef _RV_GROUP_GPU_H
#define _RV_GROUP_GPU_H

#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>


int rv_group_wrapper(int M, int C, int nsample,
    at::Tensor features_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);

void rv_group_kernel_launcher(int M, int C, int nsample,
    const float *features, const int *idx, float *out);

int rv_group_grad_wrapper(int M, int C, int nsample,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_features_tensor);

void rv_group_grad_kernel_launcher(int M, int C, int nsample,
    const float *grad_out, const int *idx, float *grad_features);

#endif
