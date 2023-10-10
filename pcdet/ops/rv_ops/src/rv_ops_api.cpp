#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "rv_assigner_gpu.h"
#include "rv_query_gpu.h"
#include "rv_group_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rv_assigner_wrapper", &rv_assigner_wrapper, "rv_assigner_wrapper");
    m.def("ball_rand_query_wrapper", &ball_rand_query_wrapper, "ball_rand_query_wrapper");
    m.def("rv_knn_query_wrapper", &rv_knn_query_wrapper, "rv_knn_query_wrapper");
    m.def("rv_fps_query_wrapper", &rv_fps_query_wrapper, "rv_fps_query_wrapper");
    m.def("rv_fps_query_wrapper_v2", &rv_fps_query_wrapper_v2, "rv_fps_query_wrapper_v2");
    m.def("rv_query_wrapper", &rv_query_wrapper, "rv_query_wrapper");
    m.def("rv_conv_query_wrapper", &rv_conv_query_wrapper, "rv_conv_query_wrapper");
    m.def("rv_rand_query_wrapper", &rv_rand_query_wrapper, "rv_rand_query_wrapper");
    m.def("rv_group_wrapper", &rv_group_wrapper, "rv_group_wrapper");
    m.def("rv_group_grad_wrapper", &rv_group_grad_wrapper, "rv_group_grad_wrapper");
}
