import torch
from torch.autograd import Function, Variable
import torch.nn as nn
from . import rv_ops_cuda
from torch.nn import functional as F
from ..pointnet2.pointnet2_stack import pointnet2_utils
import time
from ...models.model_utils.network_utils import make_fc_layers


class RangeViewAssigner(Function):

    @staticmethod
    def forward(ctx, rv_coords, rv_size, num_points_per_pixel=4):
        """
        Args:
            ctx:
            rv_coords: (N1+N2+..., 3), [bs_idx, row_idx, col_idx]
        Returns:
            rv_map: (B, rv_h, rv_w, num_points_per_pixel)
        """
        B = rv_coords[:, 0].max().int().item() + 1
        pts_num = rv_coords.shape[0]
        rv_h, rv_w = rv_size
        rv_map = -1 * rv_coords.new_ones(B, rv_h, rv_w, num_points_per_pixel).int()
        rv_map[..., 0] = 0
        rv_ops_cuda.rv_assigner_wrapper(B, pts_num, rv_h, rv_w, num_points_per_pixel, rv_coords.contiguous(), rv_map)
        return rv_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


rv_assigner = RangeViewAssigner.apply


class RangeViewQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, dilation, query_range, xyz, query_rv_xyz, query_rv_coords, rv_map, method='rv'):
        """
        Args:
            ctx:
            xyz: (N1+N2+..., 3), [x, y, z]
            query_rv_xyz: (M1+M2+..., 3), [x, y, z]
            query_rv_coords: (M1+M2+..., 3), [bs_idx, row_idx, col_idx]
            rv_map: (B, rv_h, rv_w, num_points_per_pixel)
            method: [rv, rv_balanced, rv_rand]
        Returns:
            idx: (M1+M2+..., nsample)
            empty_ball_mask: (M1+M2+...,)
        """
        pts_num = query_rv_xyz.shape[0]
        idx = torch.cuda.IntTensor(pts_num, nsample).zero_()
        sampled_pts_num = torch.cuda.IntTensor(pts_num).zero_()
        h_range, w_range = query_range
        h_dilation, w_dilation = dilation
        B, rv_h, rv_w, num_points_per_pixel = rv_map.shape
        if method == 'rv':
            rv_ops_cuda.rv_query_wrapper(B, pts_num, rv_h, rv_w, num_points_per_pixel,
                        radius, nsample, h_dilation, w_dilation, h_range, w_range,
                        xyz, query_rv_xyz, query_rv_coords, rv_map, idx, sampled_pts_num)
        else:
            rv_ops_cuda.rv_rand_query_wrapper(B, pts_num, rv_h, rv_w, num_points_per_pixel,
                        radius, nsample, h_dilation, w_dilation, h_range, w_range,
                        xyz, query_rv_xyz, query_rv_coords, rv_map, idx, sampled_pts_num)

        empty_ball_mask = sampled_pts_num == 0
        idx[empty_ball_mask] = 0

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


rv_query = RangeViewQuery.apply


def rv_assigner_v2(xyz, rv_coords, rv_size):
    rv_h, rv_w = rv_size
    keep = (rv_coords[:, 1] >= 0) & (rv_coords[:, 1] < rv_h) & (rv_coords[:, 2] >= 0) & (rv_coords[:, 2] < rv_w)
    ori_indices = keep.nonzero(as_tuple=True)[0]
    xyz, rv_coords = xyz[ori_indices], rv_coords[ori_indices]
    flatten_idx = rv_coords[:, 0] * rv_h * rv_w + rv_coords[:, 1] * rv_w + rv_coords[:, 2]
    flatten_idx, indices = torch.sort(flatten_idx)
    xyz, ori_indices = xyz[indices], ori_indices[indices]
    rv_counts = flatten_idx.bincount()
    rv_ends = rv_counts.cumsum(dim=-1)
    return xyz, ori_indices, rv_ends


class RangeViewFPSQueryV2(Function):
    @staticmethod
    def forward(ctx, radius, max_nsample, nsample, dilation, query_range, rv_size, xyz, query_rv_xyz, query_rv_coords, ori_indices, rv_ends):
        """
        Args:
            ctx:
            xyz: (N1+N2+..., 3), [x, y, z], sorted
            query_rv_xyz: (M1+M2+..., 3), [x, y, z]
            query_rv_coords: (M1+M2+..., 3), [bs_idx, row_idx, col_idx]
            rv_ends: (N,)
        Returns:
            idx: (M1+M2+..., nsample)
            empty_ball_mask: (M1+M2+...,)
        """
        B = query_rv_coords[:, 0].max().int().item() + 1
        pts_num = query_rv_xyz.shape[0]
        idx = torch.cuda.IntTensor(pts_num, nsample).zero_()
        sampled_pts_num = torch.cuda.IntTensor(pts_num).zero_()
        h_range, w_range = query_range
        h_dilation, w_dilation = dilation
        rv_h, rv_w = rv_size
        rv_length = rv_ends.shape[0]
        rv_ops_cuda.rv_fps_query_wrapper_v2(B, pts_num, rv_length, rv_h, rv_w, 
                    radius, max_nsample, nsample, h_dilation, w_dilation, h_range, w_range,
                    xyz, query_rv_xyz, query_rv_coords, rv_ends, idx, sampled_pts_num)

        empty_ball_mask = sampled_pts_num == 0
        idx[empty_ball_mask] = 0
        idx = ori_indices[idx.long()]
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


rv_fps_query_v2 = RangeViewFPSQueryV2.apply


class RangeViewConvQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, dilation, query_range, xyz, query_rv_xyz, query_rv_coords, rv_map):
        """
        Args:
            ctx:
            xyz: (N1+N2+..., 3), [x, y, z]
            query_rv_xyz: (M1+M2+..., 3), [x, y, z]
            query_rv_coords: (M1+M2+..., 3), [bs_idx, row_idx, col_idx]
            rv_map: (B, rv_h, rv_w, num_points_per_pixel)
            method: [rv, rv_balanced, rv_rand]
        Returns:
            idx: (M1+M2+..., nsample)
            empty_ball_mask: (M1+M2+..., nsample)
        """
        pts_num = query_rv_xyz.shape[0]
        idx = torch.cuda.IntTensor(pts_num, nsample).fill_(-1)
        h_range, w_range = query_range
        h_dilation, w_dilation = dilation
        B, rv_h, rv_w, num_points_per_pixel = rv_map.shape
        rv_ops_cuda.rv_conv_query_wrapper(B, pts_num, rv_h, rv_w, num_points_per_pixel,
                    radius, nsample, h_dilation, w_dilation, h_range, w_range,
                    xyz, query_rv_xyz, query_rv_coords, rv_map, idx)

        empty_ball_mask = (idx == -1)
        idx[empty_ball_mask] = 0

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


rv_conv_query = RangeViewConvQuery.apply


class RangeViewFPSQuery(Function):
    @staticmethod
    def forward(ctx, radius, max_nsample, nsample, dilation, query_range, xyz, query_rv_xyz, query_rv_coords, rv_map, method='rv'):
        """
        Args:
            ctx:
            xyz: (N1+N2+..., 3), [x, y, z]
            query_rv_xyz: (M+M+..., 3), [x, y, z]
            query_rv_coords: (M+M+..., 3), [bs_idx, row_idx, col_idx]
            rv_map: (B, rv_h, rv_w, num_points_per_pixel)
            method: [rv, rv_balanced, rv_rand]
        Returns:
            idx: (M+M+..., nsample)
            empty_ball_mask: (M+M+...,)
        """
        pts_num = query_rv_xyz.shape[0]
        idx = torch.cuda.IntTensor(pts_num, nsample).zero_()
        sampled_pts_num = torch.cuda.IntTensor(pts_num).zero_()
        h_range, w_range = query_range
        h_dilation, w_dilation = dilation
        B, rv_h, rv_w, num_points_per_pixel = rv_map.shape

        rv_ops_cuda.rv_fps_query_wrapper(B, pts_num, rv_h, rv_w, num_points_per_pixel,
                    radius, max_nsample, nsample, h_dilation, w_dilation, h_range, w_range,
                    xyz, query_rv_xyz, query_rv_coords, rv_map, idx, sampled_pts_num)

        empty_ball_mask = sampled_pts_num == 0
        idx[empty_ball_mask] = 0

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


rv_fps_query = RangeViewFPSQuery.apply


class RangeViewKNN(Function):
    @staticmethod
    def forward(ctx, nsample, radius, dilation, query_range, xyz, feats, query_rv_xyz, query_rv_feats, query_rv_coords, rv_map):
        """
        Args:
            ctx:
            xyz: (N1+N2+..., 3), [x, y, z]
            query_rv_xyz: (M1+M2+..., 3), [x, y, z]
            query_rv_feats: (M1+M2+..., C)
            query_rv_coords: (M1+M2+..., 3), [bs_idx, row_idx, col_idx]
            rv_map: (B, rv_h, rv_w, num_points_per_pixel)
            method: [rv, rv_balanced, rv_rand]
        Returns:
            idx: (M1+M2+..., nsample)
        """
        pts_num, feats_dim = query_rv_feats.shape
        idx = torch.cuda.IntTensor(pts_num, nsample).zero_()
        h_range, w_range = query_range
        h_dilation, w_dilation = dilation
        B, rv_h, rv_w, num_points_per_pixel = rv_map.shape
        
        rv_ops_cuda.rv_knn_query_wrapper(B, pts_num, feats_dim, rv_h, rv_w, num_points_per_pixel,
                    nsample, radius, h_dilation, w_dilation, h_range, w_range,
                    xyz, feats, query_rv_xyz, query_rv_feats, query_rv_coords, rv_map, idx)
        
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


rv_knn = RangeViewKNN.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius, nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt):
        """
        Args:
            ctx:
            radius: float, radius of the balls
            nsample: int, maximum number of features in the balls
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        rv_ops_cuda.ball_rand_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class RangeViewGroup(Function):

    @staticmethod
    def forward(ctx, features, idx):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        M, nsample = idx.size()
        N, C = features.size()
        output = torch.cuda.FloatTensor(M, C, nsample)

        rv_ops_cuda.rv_group_wrapper(M, C, nsample, features, idx, output)

        ctx.for_backwards = (N, idx)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        N, idx = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        rv_ops_cuda.rv_group_grad_wrapper(M, C, nsample, grad_out_data, idx,
                                          grad_features.data)
        return grad_features, None


rv_group = RangeViewGroup.apply


class RangeViewQueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, dilation, query_range, query_mod='rv_balanced', use_xyz=True, max_nsample=-1):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.dilation, self.query_range, self.radius, self.nsample, self.query_mod, self.use_xyz, self.max_nsample = dilation, query_range, radius, nsample, query_mod, use_xyz, max_nsample

    def forward(self, xyz, features, query_rv_xyz, query_rv_coords=None, rv_map=None, xyz_batch_cnt=None, new_xyz_batch_cnt=None, return_data=False):
        """
        Args:
            xyz: (N1+N2+..., 3) xyz coordinates of the features
            features: (N1+N2+..., C) tensor of features to group
            query_rv_xyz: (M1+M2+..., 3) centers of the ball query
            query_rv_coords: (M1+M2+..., 3), [bs_idx, row_idx, col_idx]
            rv_map: (B, rv_h, rv_w, num_points_per_pixel) tensor of points indices of voxels
        Returns:
            grouped_features: (M1+M2+..., C+3, nsample) or (M1+M2+..., C, nsample)
        """
        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        if self.max_nsample == -1:
            if self.query_mod == 'ball':
                assert xyz_batch_cnt is not None and new_xyz_batch_cnt is not None
                idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, query_rv_xyz, new_xyz_batch_cnt)
            else:
                assert query_rv_coords is not None and rv_map is not None
                idx, empty_ball_mask = rv_query(self.radius, self.nsample, self.dilation, self.query_range, xyz, query_rv_xyz, query_rv_coords, rv_map, self.query_mod)
        else:
            idx, empty_ball_mask = rv_fps_query(self.radius, self.max_nsample, self.nsample, self.dilation, self.query_range, xyz, query_rv_xyz, query_rv_coords, rv_map, self.query_mod)

        grouped_ori_features = rv_group(features, idx)
        if self.use_xyz:
            grouped_ori_xyz = rv_group(xyz, idx)  # (M1 + M2, 3, nsample)    
            grouped_features = torch.cat([grouped_ori_xyz - query_rv_xyz.unsqueeze(-1), grouped_ori_features], dim=1)
        grouped_features[empty_ball_mask] = 0
        if return_data:
            return grouped_features, grouped_ori_xyz, grouped_ori_features, empty_ball_mask
        else:
            return grouped_features


class RangeViewSAModuleMSG(nn.Module):
    def __init__(self,
                 radii,
                 nsamples,
                 mlp_channels,
                 dilations, 
                 query_ranges,
                 out_channels=[],
                 pool_mod='max',
                 query_mod='rv',
                 use_xyz=True,
                 max_nsamples=[-1]):
        super().__init__()

        max_nsamples = [-1] * len(radii) if max_nsamples[0] == -1 else max_nsamples
        assert len(radii) == len(nsamples) == len(mlp_channels) == len(dilations) == len(query_ranges) == len(max_nsamples)
        assert isinstance(out_channels, list) 
        assert pool_mod in ['max', 'avg']
        assert query_mod in ['rv', 'rv_rand', 'rv_balanced', 'ball']
        assert isinstance(mlp_channels, list)

        self.mlp_channels = mlp_channels
        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            dilation = dilations[i]
            query_range = query_ranges[i]
            max_nsample = max_nsamples[i]
            grouper = RangeViewQueryAndGroup(
                radius,
                nsample,
                dilation,
                query_range,
                query_mod=query_mod,
                use_xyz=use_xyz,
                max_nsample=max_nsample,
            )
            self.groupers.append(grouper)

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            if use_xyz:
                mlp_channel[0] += 3

            shared_mlps = []
            for i in range(len(mlp_channel) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_channel[i], mlp_channel[i + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_channel[i + 1]),
                    nn.SiLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.out_aggregation = None
        if len(out_channels) > 0:
            in_channel = sum([mlp[-1] for mlp in self.mlp_channels])
            mlps = [in_channel] + out_channels
            shared_mlps = []
            for k in range(len(mlps) - 1):
                shared_mlps.extend([
                    nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlps[k + 1]),
                    nn.SiLU()
                ])
            self.out_aggregation = nn.Sequential(*shared_mlps)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def _pool_features(self, features):
        """Perform feature aggregation using pooling operation.

        Args:
            features (torch.Tensor): (B, C, N, K)
                Features of locally grouped points before pooling.

        Returns:
            torch.Tensor: (B, C, N)
                Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    def forward(self, xyz, features, query_rv_xyz, query_rv_coords=None, rv_map=None, xyz_batch_cnt=None, new_xyz_batch_cnt=None, return_data=False):
        """
        Args:
            xyz: (N1+N2+..., 3) xyz coordinates of the features
            features: (N1+N2+..., C) tensor of features to group
            query_rv_xyz: (M1+M2+..., 3) centers of the ball query
            query_rv_coords: (M1+M2+..., 3), [bs_idx, row_idx, col_idx]
            rv_map: (B, rv_h, rv_w, num_points_per_pixel) tensor of points indices of voxels
        Returns:
            new_features: (M1+M2+..., C)
        """
        new_features_list = []
        grouped_ori_xyz_list, grouped_ori_features_list, empty_ball_mask_list = [], [], []
        for i in range(len(self.groupers)):
            # grouped_xyz, grouped_features, empty_ball_mask = self.groupers[i](xyz, features, query_rv_xyz, query_rv_coords, rv_map, xyz_batch_cnt, new_xyz_batch_cnt)
            # # (M1+M2+..., 3+C, nsample)
            # grouped_features = torch.cat([grouped_xyz - query_rv_xyz.unsqueeze(-1), grouped_features], dim=1)
            # grouped_features[empty_ball_mask] = 0
            grouped_features = self.groupers[i](xyz, features, query_rv_xyz, query_rv_coords, rv_map, xyz_batch_cnt, new_xyz_batch_cnt, return_data)
            if return_data:
                grouped_features, grouped_ori_xyz, grouped_ori_features, empty_ball_mask = grouped_features
                grouped_ori_xyz_list.append(grouped_ori_xyz)
                grouped_ori_features_list.append(grouped_ori_features)
                empty_ball_mask_list.append(empty_ball_mask)
            # (1, 3+C, M1+M2+..., nsample)
            new_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features = self.mlps[i](new_features)

            # (1, C, M1+M2+...,)
            new_features = self._pool_features(new_features)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)
        if self.out_aggregation is not None:
            new_features = self.out_aggregation(new_features)
        new_features = new_features.squeeze(dim=0).permute(1, 0).contiguous()
        if return_data:
            return new_features, grouped_ori_xyz_list, grouped_ori_features_list, empty_ball_mask_list
        else:
            return new_features


class GeneralMKConv(nn.Module):
    def __init__(self,
                 in_channel,
                 radii,
                 mlp_channels,
                 dilations, 
                 query_ranges):
        super().__init__()

        assert len(radii) == len(mlp_channels) == len(dilations) == len(query_ranges)
        assert isinstance(mlp_channels, list)

        self.query_ranges = query_ranges
        self.radii = radii
        self.dilations = dilations
        self.pos_mlps = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(len(mlp_channels)):
            out_channel = mlp_channels[i]
            range_h, range_w = self.query_ranges[i]
            kernel_num = (range_h * 2 + 1) * (range_w * 2 + 1)
            self.pos_mlps.append(nn.Sequential(
                nn.Conv1d(3, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm1d(in_channel),
                nn.ReLU()
            ))
            self.mlps.append(nn.Sequential(
                nn.Conv1d(in_channel * kernel_num, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            ))
            in_channel = out_channel

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, features, query_rv_xyz, query_rv_coords, rv_map):
        """
        Args:
            xyz: (N1+N2+..., 3) xyz coordinates of the features
            features: (N1+N2+..., C) tensor of features to group
            query_rv_xyz: (M1+M2+..., 3) centers of the ball query
            query_rv_coords: (M1+M2+..., 3), [bs_idx, row_idx, col_idx]
            rv_map: (B, rv_h, rv_w, num_points_per_pixel) tensor of points indices of voxels
        Returns:
            new_features: (M1+M2+..., C)
        """
        new_features = features

        for i in range(len(self.mlps)):
            query_range = self.query_ranges[i]
            nsample = (query_range[0] * 2 + 1) * (query_range[1] * 2 + 1)
            idx, empty_ball_mask = rv_conv_query(
                self.radii[i], 
                nsample, 
                self.dilations[i], 
                query_range, 
                xyz.contiguous(), 
                query_rv_xyz.contiguous(), 
                query_rv_coords.int().contiguous(), 
                rv_map.contiguous()
            )
            idx = idx.long()
            grouped_xyz = xyz[idx]  # (M1+M2..., nsample, 3)
            grouped_xyz = grouped_xyz - xyz.unsqueeze(1)
            grouped_xyz = self.pos_mlps[i](grouped_xyz.permute(0, 2, 1)).permute(0, 2, 1)  # (M1+M2..., nsample, C)

            grouped_features = new_features[idx] + grouped_xyz  # (M1+M2..., nsample, C)
            grouped_features = grouped_features * (empty_ball_mask == 0).unsqueeze(-1)
            grouped_features = torch.flatten(grouped_features, start_dim=1).unsqueeze(-1)
            grouped_features = self.mlps[i](grouped_features).squeeze(-1)
            new_features = grouped_features

        return new_features


class RangeViewKNNSAModuleMSG(nn.Module):
    def __init__(self,
                 radii,
                 nsamples,
                 mlp_channels,
                 dilations, 
                 query_ranges,
                 out_channels=[[]],
                 dis_mod='feature',
                 use_xyz=True):
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlp_channels) == len(dilations) == len(query_ranges) == len(out_channels)
        assert dis_mod in ['feature', 'geometry']

        self.nsamples = nsamples
        self.radii = radii
        self.dilations = dilations
        self.query_ranges = query_ranges
        self.dis_mod = dis_mod
        self.use_xyz = use_xyz

        self.mlps = nn.ModuleList()
        for i in range(len(mlp_channels)):
            mlp_channel = mlp_channels[i]
            if use_xyz:
                mlp_channel[0] += 3
            mlp_channel[0] *= 2  # EdgeConv
            shared_mlps = []
            for k in range(len(mlp_channel) - 1):
                shared_mlps.extend([
                    nn.Conv1d(mlp_channel[k], mlp_channel[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_channel[k + 1]),
                    nn.ReLU()
                ])
            if len(shared_mlps):
                self.mlps.append(nn.Sequential(*shared_mlps))

        self.out_aggregation = nn.ModuleList()
        if len(out_channels) > 0:
            for i in range(len(out_channels)):
                if len(out_channels[i]) > 0:
                    in_channel = mlp_channels[i][-1]
                    mlps = [in_channel] + out_channels[i]
                    shared_mlps = []
                    for k in range(len(mlps) - 1):
                        shared_mlps.extend([
                            nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                            nn.BatchNorm1d(mlps[k + 1]),
                            nn.ReLU()
                        ])
                    self.out_aggregation.append(nn.Sequential(*shared_mlps))
                else:
                    self.out_aggregation.append(None)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, feats, query_rv_xyz, query_rv_feats, query_rv_coords, rv_map):
        for i in range(len(self.mlps)):
            # TODO: need to be general
            src_knn_metric = xyz if self.dis_mod == 'geometry' else feats
            dst_knn_metric = query_rv_xyz if self.dis_mod == 'geometry' else query_rv_feats
            k = self.nsamples[i]
            knn_idx, knn_empty = rv_knn(
                k,
                self.radii[i],
                self.dilations[i],
                self.query_ranges[i],
                xyz,
                src_knn_metric,
                query_rv_xyz,
                dst_knn_metric,
                query_rv_coords.int(), 
                rv_map
            )
            if self.use_xyz:
                x = torch.cat([xyz, feats], dim=-1)
            else:
                x = feats
            # (M1+M2+..., nsample, C) -> (M1+M2+..., C, nsample)
            s = x[knn_idx.long()].permute(0, 2, 1)
            """ EdgeConv """
            x = x.unsqueeze(2).repeat(1, 1, k)
            x = torch.cat([s - x, x], dim=1)
            x[knn_empty] = 0
            x = torch.max(self.mlps[i](x), dim=-1, keepdim=True)[0]
            if self.out_aggregation[i] is not None:
                x = self.out_aggregation[i](x)
            x = x.squeeze(-1)
            feats = x
            query_rv_feats = x

        return feats


class DynamicKNNEncoder(nn.Module):
    def __init__(self, in_channels, d_models, dim_feedforwards, radii, nsamples, dilations,
                 query_ranges, nheads, dis_mod='feature', use_xyz=True):
        super().__init__()
        self.dknn_layers = nn.ModuleList()
        for i in range(len(radii)):
            self.dknn_layers.append(DynamicKNNLayer(
                in_channels, d_models[i], dim_feedforwards[i], radii[i], nsamples[i],
                dilations[i], query_ranges[i], nheads[i], dis_mod, use_xyz
            ))
            in_channels = [d_models[i], d_models[i]]

    def forward(self, src_xyz, src_feats, src_rv_coords, rv_map):
        for i in range(len(self.dknn_layers)):
            src_feats = self.dknn_layers[i](
                src_xyz, src_feats, src_xyz, src_feats, src_rv_coords, rv_map
            )
        return src_feats


class DynamicKNNDecoder(nn.Module):
    def __init__(self, in_channels, d_models, dim_feedforwards, radii, nsamples, dilations,
                 query_ranges, nheads, dis_mod='feature', use_xyz=True):
        super().__init__()
        self.dknn_layers = nn.ModuleList()
        for i in range(len(radii)):
            self.dknn_layers.append(DynamicKNNLayer(
                in_channels, d_models[i], dim_feedforwards[i], radii[i], nsamples[i],
                dilations[i], query_ranges[i], nheads[i], dis_mod, use_xyz
            ))
            in_channels = [in_channels[0], d_models[i]]

    def forward(self, src_xyz, src_feats, dst_xyz, dst_feats, dst_rv_coords, rv_map):
        for i in range(len(self.dknn_layers)):
            dst_feats = self.dknn_layers[i](
                src_xyz, src_feats, dst_xyz, dst_feats, dst_rv_coords, rv_map
            )
        return dst_feats


class DynamicKNNLayer(nn.Module):
    def __init__(self, in_channels, d_model, dim_feedforward, radius, nsample, dilation,
                 query_range, nhead=1, dis_mod='feature', use_xyz=True):
        super().__init__()

        assert dis_mod in ['feature', 'geometry']

        self.nhead, self.d_model, self.radius, self.nsample, \
            self.dilation, self.query_range, self.use_xyz, self.dis_mod = \
            nhead, d_model, radius, nsample, dilation, query_range, use_xyz, dis_mod

        self.fc_q = nn.Linear(in_channels[1], d_model)
        self.fc_k = nn.Linear(in_channels[0], d_model)

        self.mlps = nn.ModuleList()
        for i in range(self.nhead):
            dim_split = self.d_model // self.nhead
            if use_xyz:
                dim_split += 3
            dim_split *= 2
            mlps = []
            for j in range(len(dim_feedforward)):
                mlps.extend([
                    nn.Conv1d(dim_split, dim_feedforward[j], kernel_size=1, bias=False),
                    nn.BatchNorm1d(dim_feedforward[j]),
                    nn.ReLU()
                ])
                dim_split = dim_feedforward[j]
            self.mlps.append(nn.Sequential(*mlps))

        self.out_conv = nn.Sequential(
            nn.Conv1d(self.nhead * dim_feedforward[-1], d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, src_xyz, src_feats, dst_xyz, dst_feats, dst_rv_coords, rv_map):
        dim_split = self.d_model // self.nhead
        Q_split = torch.split(self.fc_q(dst_feats), dim_split, dim=-1)  # [(N, C1), (N, C2), ...]
        K_split = torch.split(self.fc_k(src_feats), dim_split, dim=-1)  # [(N, C1), (N, C2), ...]
        out_feats = []
        for i in range(self.nhead):
            cur_Q, cur_K = Q_split[i].contiguous(), K_split[i].contiguous()
            src_knn_metric = src_xyz if self.dis_mod == 'geometry' else cur_K
            dst_knn_metric = dst_xyz if self.dis_mod == 'geometry' else cur_Q
            # knn_idx, knn_empty = rv_knn(
            #     self.nsample, self.radius, self.dilation, self.query_range,
            #     src_xyz, src_knn_metric, dst_xyz, dst_knn_metric, dst_rv_coords.int(), rv_map
            # )
            knn_idx, knn_empty = rv_query(self.radius, self.nsample, self.dilation, self.query_range, \
                src_xyz, dst_xyz, dst_rv_coords.int(), rv_map, 'rv_rand')
            if self.use_xyz:
                cur_K = torch.cat([cur_K, src_xyz], dim=1)
                cur_Q = torch.cat([cur_Q, dst_xyz], dim=1)
            # (M1+M2+..., nsample, C) -> (M1+M2+..., C, nsample)
            cur_K = cur_K[knn_idx.long()].permute(0, 2, 1)
            cur_Q = cur_Q.unsqueeze(2).repeat(1, 1, self.nsample)

            x = torch.cat([cur_K - cur_Q, cur_Q], dim=1)
            x[knn_empty] = 0
            out_feats.append(torch.max(self.mlps[i](x), dim=-1, keepdim=True)[0])
        # (N, C1+C2+...)
        out_feats = self.out_conv(torch.cat(out_feats, dim=1)).squeeze(-1)
        return out_feats


class RVSAEncoder(nn.Module):
    def __init__(self, in_channels, d_ins, d_hiddens, d_outs, radii, nsamples, dilations,
                 query_ranges):
        super().__init__()
        self.sa_layers = nn.ModuleList()
        sum_c = 0
        for i in range(len(radii)):
            self.sa_layers.append(RVSALayer(
                in_channels, d_ins[i], d_hiddens[i], radii[i], nsamples[i],
                dilations[i], query_ranges[i]
            ))
            sum_c += d_hiddens[i][-1]
        self.out_conv = make_fc_layers(d_outs, sum_c, linear=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, src_xyz, src_feats, dst_xyz, dst_rv_coords, rv_map):
        out_feats = []
        for i in range(len(self.sa_layers)):
            dst_feats = self.sa_layers[i](
                src_xyz, src_feats, dst_xyz, dst_rv_coords, rv_map
            )
            out_feats.append(dst_feats)
        out_feats = torch.cat(out_feats, dim=1)
        out_feats = self.out_conv(out_feats).squeeze(-1)
        return out_feats


class RVSALayer(nn.Module):
    def __init__(self, in_channels, d_in, d_hidden, radius, nsample, dilation, query_range):
        super().__init__()

        self.d_in, self.radius, self.nsample, self.dilation, self.query_range = \
            d_in, radius, nsample, dilation, query_range
        self.fc_i = nn.Linear(in_channels, d_in)
        self.p_mlp = nn.Sequential(
            nn.Conv1d(6, d_in, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_in),
            nn.SiLU()
        )
        self.mlp = make_fc_layers(d_hidden, d_in, linear=False)

    def forward(self, src_xyz, src_feats, dst_xyz, dst_rv_coords, rv_map):
        cur_i = self.fc_i(src_feats)  # (N, C)
        idx, empty = rv_query(self.radius, self.nsample, self.dilation, self.query_range, \
            src_xyz, dst_xyz, dst_rv_coords.int(), rv_map, 'rv_rand')
        src_p = src_xyz[idx.long()].permute(0, 2, 1)
        dst_p = dst_xyz.unsqueeze(2).repeat(1, 1, self.nsample)
        off_p = self.p_mlp(torch.cat([src_p - dst_p, dst_p], dim=1))
        src_f = cur_i[idx.long()].permute(0, 2, 1)
        x = off_p + src_f
        x[empty] = 0
        x = torch.max(self.mlp(x), dim=-1, keepdim=True)[0]
        return x
