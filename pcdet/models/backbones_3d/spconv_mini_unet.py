from functools import partial
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from .spconv_backbone import SparseBasicBlock, post_act_block
import pickle


class SparseMiddleLayer(spconv.SparseModule):
    def __init__(self, in_channel):
        super().__init__()

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            # [200, 176]
            block(in_channel, 128, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='_spconv1', conv_type='spconv', dim=2),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='_subm1', dim=2),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='_subm1', dim=2),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='_subm1', dim=2),
            # block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='_subm1', dim=2),
        )
        self.conv2 = spconv.SparseSequential(
            # [100, 88]
            block(128, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='_spconv2', conv_type='spconv', dim=2),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='_subm2', dim=2),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='_subm2', dim=2),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='_subm2', dim=2),
            # block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='_subm2', dim=2),
        )
        self.inv_conv1 = block(128, 128, 3, norm_fn=norm_fn, indice_key='_subm1', dim=2)
        self.inv_conv2 = block(128, 128, 3, norm_fn=norm_fn, indice_key='_spconv2', conv_type='inverseconv', dim=2)

        self.conv_out = block(256, 128, 3, norm_fn=norm_fn, indice_key='_subm1', dim=2)

    def forward(self, x):
        from spconv.pytorch import SparseConvTensor
        # 1) grab the old tensor
        old = x  

        # 2) compute your sort order
        sorted_idx = torch.argsort(old.indices[:, 0])

        # 3) slice out the sorted pieces
        new_indices  = old.indices[sorted_idx]
        new_features = old.features[sorted_idx]

        # 4) re-instantiate a new SparseConvTensor
        x = SparseConvTensor(
            new_features,
            new_indices,
            old.spatial_shape,    # same spatial dims
            old.batch_size,       # same batch size
        )
        x_in = x.dense()
        N, _, _, Y, X = x_in.shape
        x_in = spconv.SparseConvTensor.from_dense(x_in.view(N, -1, Y, X).permute(0, 2, 3, 1).contiguous())
        x1 = self.conv1(x_in)
        x2 = self.conv2(x1)
        x1_up = self.inv_conv1(x1)
        x2_up = self.inv_conv2(x2)
        x_out = self.conv_out(replace_feature(x1_up, torch.cat([x1_up.features, x2_up.features], dim=-1)))
        slices = [x.indices[:, i].long() for i in [0, 2, 3]]
        # (B, C, Y, X) -> (B, Y, X, C)
        return replace_feature(x, x.features + x_out.dense().permute(0, 2, 3, 1)[slices]), x_out


class MiniUNet(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            block(128, 128, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='spconv5', conv_type='spconv')
        )

        self.middle_conv = SparseMiddleLayer(256)  # [200, 176, 2] -> [200, 176, 2]
        # self.middle_conv = SparseMiddleLayer(128 * 5)  # [200, 176, 5] -> [200, 176, 5]

        # decoder
        # [200, 176, 5] <- [200, 176, 2]
        self.conv_up_t5 = block(128, 128, 3, norm_fn=norm_fn, indice_key='subm5')
        # self.conv_up_t5 = SparseBasicBlock(128, 128, indice_key='subm5', norm_fn=norm_fn)
        self.inv_conv5 = block(128, 128, (3, 1, 1), norm_fn=norm_fn, indice_key='spconv5', conv_type='inverseconv')
        self.conv_up_m5 = block(256, 128, 3, norm_fn=norm_fn, indice_key='subm5')

        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='subm4')
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.inv_conv4 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, indice_key='subm4')

        # [800, 704, 21] <- [400, 352, 11]
        # self.conv_up_t3 = block(32, 32, 3, norm_fn=norm_fn, indice_key='subm3')
        # self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        # self.conv_up_m3 = block(64, 64, 3, norm_fn=norm_fn, indice_key='subm3')

        self.num_point_features = 64

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x_up = conv_inv(x_bottom)
        x = replace_feature(x_trans, torch.cat((x_trans.features, x_up.features), dim=-1))
        x = conv_m(x)
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        x_conv5, x_bev = self.middle_conv(x_conv5)
        # x_conv4, x_bev = self.middle_conv(x_conv4)

        # for segmentation head
        # [200, 176, 5] <- [200, 176, 2]
        x_up5 = self.UR_block_forward(x_conv4, x_conv5, self.conv_up_t5, self.conv_up_m5, self.inv_conv5)
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv3, x_up5, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # x_up4 = self.UR_block_forward(x_conv3, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        # x_up3 = self.UR_block_forward(x_conv2, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)

        batch_dict.update({
            'encoded_spconv_tensor': x_up4,
            'encoded_spconv_tensor_stride': 4
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_up4': x_up4,
                'x_bev': x_bev,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_up4': 4,
                'x_bev': 8,
            }
        })

        return batch_dict


class MiniUNetV1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            block(64, 128, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='spconv5', conv_type='spconv')
        )

        self.middle_conv = SparseMiddleLayer(256)  # [200, 176, 2] -> [200, 176, 2]

        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        _, x_bev = self.middle_conv(x_conv5)

        batch_dict.update({
            'encoded_spconv_tensor': x_bev,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_bev': x_bev,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_bev': 8,
            }
        })

        return batch_dict