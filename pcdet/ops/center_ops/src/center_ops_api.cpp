#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int draw_center_gpu(at::Tensor gt_boxes_tensor, at::Tensor heatmap_tensor, at::Tensor gt_ind_tensor,
                        at::Tensor gt_mask_tensor, at::Tensor gt_cat_tensor,
                        at::Tensor gt_box_encoding_tensor, at::Tensor gt_cnt_tensor,
                        int min_radius, float out_factor, float gaussian_overlap);

int draw_bev_all_gpu(at::Tensor gt_boxes_tensor, at::Tensor gt_corners_tensor,
                     at::Tensor center_map_tensor, at::Tensor corner_map_tensor,
                     int min_radius, float voxel_x, float voxel_y, float range_x, float range_y,
                     float out_factor, float gaussian_overlap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("draw_center_gpu", &draw_center_gpu, "centerpoint assignment creation");
  m.def("draw_bev_all_gpu", &draw_bev_all_gpu, "center corner assignment creation");
}