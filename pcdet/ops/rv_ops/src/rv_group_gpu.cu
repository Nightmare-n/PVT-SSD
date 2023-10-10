#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "rv_group_gpu.h"


__global__ void rv_group_grad_kernel(int M, int C, int nsample,
    const float *grad_out, const int *idx, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    int C_idx = (index / nsample) % C;
    int pt_idx = (index / nsample / C);

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

    grad_out += pt_idx * C * nsample + C_idx * nsample + sample_idx;
    idx += pt_idx * nsample + sample_idx;
    grad_features += idx[0] * C + C_idx;

    atomicAdd(grad_features, grad_out[0]);
}

void rv_group_grad_kernel_launcher(int M, int C, int nsample,
    const float *grad_out, const int *idx, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features

    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    rv_group_grad_kernel<<<blocks, threads>>>(M, C, nsample, grad_out, idx, grad_features);
}

__global__ void rv_group_kernel(int M, int C, int nsample,
                                const float *features, const int *idx, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    int C_idx = (index / nsample) % C;
    int pt_idx = (index / nsample / C);

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

    idx += pt_idx * nsample + sample_idx;
    int in_idx = idx[0] * C + C_idx;
    int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx;

    out[out_idx] = features[in_idx];
}

void rv_group_kernel_launcher(int M, int C, int nsample,
    const float *features, const int *idx, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor

    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    rv_group_kernel<<<blocks, threads>>>(M, C, nsample, features, idx, out);
}
