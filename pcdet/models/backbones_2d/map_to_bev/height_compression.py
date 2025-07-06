
import torch
import torch.nn as nn
import torch_scatter
from ....utils.spconv_utils import spconv


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.type = self.model_cfg.get('TYPE', 'cat')
        self.to_sparse = self.model_cfg.get('TO_SPARSE', False)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        from spconv.pytorch import SparseConvTensor
        
        old: SparseConvTensor = encoded_spconv_tensor

        # 1) compute sort order
        sorted_idx = torch.argsort(old.indices[:, 0])

        # 2) slice out your sorted pieces
        new_indices  = old.indices[sorted_idx]
        new_features = old.features[sorted_idx]

        # 3) re-build a brand-new tensor
        new = SparseConvTensor(
            new_features,
            new_indices,
            old.spatial_shape,   # same spatial dims
            old.batch_size       # same batch size
        )
        batch_dict['encoded_spconv_tensor'] = new
        encoded_spconv_tensor = new
        spatial_features = encoded_spconv_tensor.dense()
        N, H, W = spatial_features.shape[0], spatial_features.shape[-2], spatial_features.shape[-1]

        if self.type == 'cat':
            spatial_features = spatial_features.view(N, -1, H, W)
        else:
            indices = encoded_spconv_tensor.indices.long()
            flat_indices = indices[:, 0] * H * W + indices[:, 2] * W + indices[:, 3]

            if self.type == 'max':
                spatial_features = torch_scatter.scatter(encoded_spconv_tensor.features, flat_indices, dim=0, dim_size=N*H*W, reduce='max')
            elif self.type == 'avg':
                spatial_features = torch_scatter.scatter(encoded_spconv_tensor.features, flat_indices, dim=0, dim_size=N*H*W, reduce='mean')
            elif self.type == 'max_avg':
                spatial_features = torch.cat([
                    torch_scatter.scatter(encoded_spconv_tensor.features, flat_indices, dim=0, dim_size=N*H*W, reduce='max'),
                    torch_scatter.scatter(encoded_spconv_tensor.features, flat_indices, dim=0, dim_size=N*H*W, reduce='mean')
                ], dim=-1)
            else:
                raise NotImplementedError
            spatial_features = spatial_features.view(N, H, W, -1).permute(0, 3, 1, 2)  # (N, C, H, W)

        if self.to_sparse:
            spatial_features = spatial_features.permute(0, 2, 3, 1).contiguous()
            spatial_features = spconv.SparseConvTensor.from_dense(spatial_features)

        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
