import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from ..model_utils.network_utils import make_fc_layers
from .point_head_template import PointHeadTemplate
from ...ops.pointnet2.pointnet2_batch.pointnet2_utils import point_sampler
from ...ops.rv_ops import rv_ops_utils
from ...ops.pointnet2.pointnet2_stack.voxel_query_utils import voxel_knn_query
from ...ops.pointnet2.pointnet2_stack.pointnet2_utils import k_interpolate
from ...utils.rv_utils import generate_rv_coords
from ..model_utils.transformer import Transformer
from ...ops.center_ops import center_ops_cuda
from ..backbones_3d.spconv_backbone import post_act_block
from ...utils.spconv_utils import spconv
from functools import partial
from spconv.pytorch import SparseConvTensor

class VoteLayer(nn.Module):
    def __init__(self, offset_range, input_channels, mlps, num_class=1):
        super().__init__()
        self.offset_range = offset_range
        self.offset_conv = make_fc_layers(mlps, input_channels, 3, linear=True)
        self.cls_conv = make_fc_layers(mlps, input_channels, num_class, linear=True)
        # self.cls_conv = make_fc_layers(mlps, input_channels, 1, linear=True)
        self.cls_conv[-1].bias.data.fill_(-2.19)

    def forward(self, seeds, seed_feats):
        """
        Args:
            seeds: (N, 4), [bs_idx, x, y, z]
            features: (N, C)
        Return:
            new_xyz: (N, 3)
        """
        seed_offset = self.offset_conv(seed_feats)  # (N, 3)
        seed_cls = self.cls_conv(seed_feats)  # (N, num_class)
        limited_offset = []
        for axis in range(len(self.offset_range)):
            limited_offset.append(seed_offset[..., axis].clamp(
                min=-self.offset_range[axis],
                max=self.offset_range[axis]))
        limited_offset = torch.stack(limited_offset, dim=-1)
        votes = seeds[:, 1:4] + limited_offset
        votes = torch.cat([seeds[:, 0:1], votes], dim=-1)
        return votes, seed_cls, seed_offset


def fps_pool_layer(points, point_feats, point_scores, batch_size, model_cfg, mode):
    fps_indices = []
    pre_sum = 0
    for bs_idx in range(batch_size):
        cur_points, cur_point_feats, cur_point_scores = \
            points[points[:, 0] == bs_idx][:, 1:4], point_feats[points[:, 0] == bs_idx], \
                point_scores[points[:, 0] == bs_idx]
        assert len(cur_points) == len(cur_point_feats) == len(cur_point_scores)
        topk_nponits = min(len(cur_point_scores), model_cfg.MAX_NPOINTS[mode])
        _, topk_indices = torch.topk(cur_point_scores.sigmoid(), topk_nponits, dim=0)
        
        cur_points, cur_point_feats, cur_point_scores = \
            cur_points[topk_indices], cur_point_feats[topk_indices], \
                cur_point_scores[topk_indices]
        
        cur_fps_indices = []
        for fps_npoints, fps_type in zip(model_cfg.NPOINTS[mode], model_cfg.TYPE[mode]):
            if fps_npoints > 0:
                cur_fps_indices_ = point_sampler(
                    fps_type=fps_type, 
                    xyz=cur_points.unsqueeze(0).contiguous(), 
                    npoints=fps_npoints, 
                    features=cur_point_feats.unsqueeze(0).transpose(1, 2).contiguous(),
                    scores=cur_point_scores.unsqueeze(0).contiguous()
                ).squeeze(0)
            else:
                cur_fps_indices_ = torch.arange(len(cur_points)).to(cur_points.device)
            cur_fps_indices.append(cur_fps_indices_)
        cur_fps_indices = torch.cat(cur_fps_indices, dim=0)

        cur_fps_indices = topk_indices[cur_fps_indices.long()]
        fps_indices.append(cur_fps_indices + pre_sum)
        pre_sum += torch.sum(points[:, 0] == bs_idx)
    fps_indices = torch.cat(fps_indices, dim=0).long()
    return fps_indices


class PVTSSDHead(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, voxel_size, point_cloud_range, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.dense_conv2d = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            # nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block
        self.exp = spconv.SparseSequential(
            block(input_channels, input_channels, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='sp_exp', conv_type='spconv', dim=2),
        )
        self.vote_layer = VoteLayer(
            model_cfg.VOTE_CONFIG.OFFSET_RANGE,
            input_channels,
            model_cfg.VOTE_CONFIG.MLPS,
            num_class
        )

        self.vote_reduce_conv = make_fc_layers(
            [128],
            input_channels + 128,
            linear=True
        )
        
        self.point_knn_cfg = model_cfg.POINT_KNN_CONFIG
        in_channels = 0
        for k, src_name in enumerate(self.point_knn_cfg.FEATURES_SOURCE):
            layer_cfg = self.point_knn_cfg.POOL_LAYERS
            in_channels += layer_cfg[src_name].DIM

        self.point_feat_reduction = make_fc_layers(
            [128],
            in_channels,
            linear=True
        )
        
        trans_cfg = model_cfg.PV_TRANS_CONFIG
        self.pv_transformer = Transformer(
            d_model=128, nhead=trans_cfg.NHEAD, num_decoder_layers=trans_cfg.NUM_DEC, dim_feedforward=trans_cfg.FNN_DIM,
            dropout=trans_cfg.DP_RATIO 
        )

        self.shared_conv = self.make_fc_layers(
            fc_cfg=self.model_cfg.SHARED_FC,
            input_channels=128,
            linear=True
        )
        channel_out = self.model_cfg.SHARED_FC[-1]
        self.cls_conv = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=channel_out,
            output_channels=num_class,
            linear=True
        )
        self.box_conv = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=channel_out,
            output_channels=self.box_coder.code_size,
            linear=True
        )

        self._reset_parameters(weight_init='xavier')

    def _reset_parameters(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_losses(self, losses_cfg):
        cls_loss_type = 'WeightedBinaryCrossEntropyLoss' \
            if losses_cfg.CLS_LOSS.startswith('WeightedBinaryCrossEntropyLoss') else losses_cfg.CLS_LOSS
        self.cls_loss_func = getattr(loss_utils, cls_loss_type)(
            **losses_cfg.get('CLS_LOSS_CONFIG', {})
        )

        reg_loss_type = losses_cfg.REG_LOSS
        self.reg_loss_func = getattr(loss_utils, reg_loss_type)(
            code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
            **losses_cfg.get('REG_LOSS_CONFIG', {})
        )

        aux_cls_loss_type = losses_cfg.get('AUX_CLS_LOSS', None)
        if aux_cls_loss_type is not None:
            self.aux_cls_loss_func = getattr(loss_utils, aux_cls_loss_type)(
                **losses_cfg.get('AUX_CLS_LOSS_CONFIG', {})
            )
        
        self.seed_cls_loss_func = loss_utils.FocalLossCenterNet()
        self.seed_reg_loss_func = loss_utils.WeightedSmoothL1Loss()

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        gt_boxes = input_dict['gt_boxes']
        batch_size = gt_boxes.shape[0]

        """ Aux loss """
        spatial_features = input_dict['spatial_features']
        
        sorted_idx = torch.argsort(spatial_features.indices[:, 0])
        new_idx = spatial_features.indices[sorted_idx]
        new_features = spatial_features.features[sorted_idx]
        new = SparseConvTensor(
            new_features,
            new_idx,
            spatial_features.spatial_shape,   # same spatial dims
            spatial_features.batch_size       # same batch size
        )
        input_dict['spatial_features'] = new
        spatial_features = new

        spatial_features_stride = input_dict['spatial_features_stride']
        feature_map_size = spatial_features.spatial_shape
        feature_map_stride = spatial_features_stride

        gt_corners = box_utils.boxes_to_corners_3d(gt_boxes.view(-1, gt_boxes.shape[-1]))
        gt_corners = gt_corners[:, :4, :2].contiguous().view(batch_size, -1, 4, 2)
        center_map = torch.zeros((batch_size, self.num_class, feature_map_size[0], feature_map_size[1]),
                                 dtype=torch.float32).to(gt_boxes.device)
        corner_map = torch.zeros((batch_size, self.num_class, 4, feature_map_size[0], feature_map_size[1]),
                                 dtype=torch.float32).to(gt_boxes.device)
        center_ops_cuda.draw_bev_all_gpu(gt_boxes, gt_corners, center_map, corner_map, self.model_cfg.TARGET_CONFIG.MIN_RADIUS,
                                         self.voxel_size[0], self.voxel_size[1],
                                         self.point_cloud_range[0], self.point_cloud_range[1],
                                         feature_map_stride, self.model_cfg.TARGET_CONFIG.GAUSSIAN_OVERLAP)

        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        
        central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)
        vote_targets_dict = self.assign_stack_targets(
            points=input_dict['votes'], gt_boxes=gt_boxes, 
            set_ignore_flag=False, use_ball_constraint=True,
            ret_part_labels=False, ret_box_labels=True, central_radius=central_radius
        )

        seed_targets_dict = {
            'seed_cls_labels_list': [],
            'seed_cls_targets_list': [],
            'gt_box_of_fg_seeds_list': []
        }
        assert len(input_dict['seeds_list']) == 1
        for i, seeds in enumerate(input_dict['seeds_list']):
            cur_seed_targets_dict = self.assign_stack_targets(
                points=seeds, gt_boxes=extend_gt_boxes,
                # points=seeds, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=False, use_ball_constraint=False,
                # set_ignore_flag=True, use_ball_constraint=False,
                ret_part_labels=False, ret_box_labels=False,
                # use_topk=True, topk=1, dim=2
            )
            seed_targets_dict['seed_cls_labels_list'].append(cur_seed_targets_dict['point_cls_labels'])
            seed_targets_dict['gt_box_of_fg_seeds_list'].append(cur_seed_targets_dict['gt_box_of_fg_points'])

        x_bev_coords = spatial_features.indices.long()
        seed_targets_dict['seed_cls_targets_list'].append(
            center_map[x_bev_coords[:, 0], :, x_bev_coords[:, 1], x_bev_coords[:, 2]]
        )

        targets_dict = {
            'vote_cls_labels': vote_targets_dict['point_cls_labels'],
            'vote_box_labels': vote_targets_dict['point_box_labels'],
            'gt_box_of_fg_votes': vote_targets_dict['gt_box_of_fg_points'],
            'seed_cls_labels_list': seed_targets_dict['seed_cls_labels_list'],
            'seed_cls_targets_list': seed_targets_dict['seed_cls_targets_list'],
            'gt_box_of_fg_seeds_list': seed_targets_dict['gt_box_of_fg_seeds_list'],
        }
        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        seed_reg_loss, tb_dict = self.get_seed_reg_loss(tb_dict)
        seed_cls_loss, tb_dict = self.get_seed_cls_loss(tb_dict)
        vote_cls_loss, tb_dict = self.get_vote_cls_loss(tb_dict)
        vote_reg_loss, tb_dict = self.get_vote_reg_loss(tb_dict)
        vote_corner_loss, tb_dict = self.get_vote_corner_loss(tb_dict)
        point_loss = seed_reg_loss + seed_cls_loss + vote_cls_loss + vote_reg_loss + vote_corner_loss
        return point_loss, tb_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)
        return y

    def get_seed_single_reg_loss(self, votes, seed_cls_labels, gt_box_of_fg_seeds, index, tb_dict=None):
        pos_mask = seed_cls_labels > 0
        seed_center_labels = gt_box_of_fg_seeds[:, 0:3]
        seed_center_loss = self.seed_reg_loss_func(
            votes[pos_mask][:, 1:], seed_center_labels
        ).sum(dim=-1).mean()
        seed_center_loss = seed_center_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['seed_reg_weight_list'][index]

        tb_dict.update({
            f'seed_reg_loss_{index}': seed_center_loss.item(),
            f'seed_pos_num_{index}': int(pos_mask.sum().item() / self.forward_ret_dict['batch_size'])
        })
        return seed_center_loss, tb_dict

    def get_seed_reg_loss(self, tb_dict=None):
        seed_cls_labels_list = self.forward_ret_dict['seed_cls_labels_list']
        gt_box_of_fg_seeds_list = self.forward_ret_dict['gt_box_of_fg_seeds_list']
        votes_list = self.forward_ret_dict['votes_list']
        seed_center_loss_list = []
        for i in range(len(votes_list)):
            seed_center_loss, tb_dict = self.get_seed_single_reg_loss(
                votes_list[i],
                seed_cls_labels_list[i],
                gt_box_of_fg_seeds_list[i],
                i,
                tb_dict
            )
            seed_center_loss_list.append(seed_center_loss)
        return sum(seed_center_loss_list), tb_dict

    def get_seed_single_cls_loss(self, point_cls_preds, point_cls_labels, index, tb_dict=None):

        if self.model_cfg.TARGET_CONFIG.SEED_CLS == 'center':
            assert len(point_cls_preds) == len(point_cls_labels)
            point_loss_cls = self.seed_cls_loss_func(self.sigmoid(point_cls_preds), point_cls_labels)
        else:
            positives = point_cls_labels > 0
            negatives = point_cls_labels == 0
            cls_weights = negatives * 1.0 + positives * 1.0
            pos_normalizer = positives.float().sum() if self.model_cfg.LOSS_CONFIG.AUX_CLS_POS_NORM else cls_weights.sum()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            num_class = 1
            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels > 0).unsqueeze(-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]
            cls_loss_src = self.aux_cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
            point_loss_cls = cls_loss_src.sum()

        point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['seed_cls_weight_list'][index]
        tb_dict.update({
            f'seed_cls_loss_{index}': point_loss_cls.item(),
        })
        return point_loss_cls, tb_dict

    def get_seed_cls_loss(self, tb_dict=None):
        seed_cls_labels_list = self.forward_ret_dict['seed_cls_targets_list'] if self.model_cfg.TARGET_CONFIG.SEED_CLS == 'center' else self.forward_ret_dict['seed_cls_labels_list']
        seeds_cls_list = self.forward_ret_dict['seeds_cls_list']
        seed_cls_loss_list = []
        for i in range(len(seeds_cls_list)):
            seed_cls_loss, tb_dict = self.get_seed_single_cls_loss(
                seeds_cls_list[i],
                seed_cls_labels_list[i],
                i,
                tb_dict
            )
            seed_cls_loss_list.append(seed_cls_loss)
        return sum(seed_cls_loss_list), tb_dict

    def get_vote_cls_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['vote_cls_labels']
        point_cls_preds = self.forward_ret_dict['vote_cls_preds']

        positives = point_cls_labels > 0
        negatives = point_cls_labels == 0
        cls_weights = negatives * 1.0 + positives * 1.0
        pos_normalizer = positives.float().sum() if self.model_cfg.LOSS_CONFIG.CLS_POS_NORM else cls_weights.sum()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]

        if 'WithCenterness' in self.model_cfg.LOSS_CONFIG.CLS_LOSS:
            votes = self.forward_ret_dict['votes'].detach()
            gt_box_of_fg_votes = self.forward_ret_dict['gt_box_of_fg_votes']
            pos_centerness = box_utils.generate_centerness_mask(votes[positives][:, 1:], gt_box_of_fg_votes)
            centerness_mask = positives.new_zeros(positives.shape).float()
            centerness_mask[positives] = pos_centerness
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1)

        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        point_loss_cls = point_loss_cls * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_cls_weight']
        tb_dict.update({
            'vote_cls_loss': point_loss_cls.item(),
            'vote_pos_num': int(positives.sum().item() / self.forward_ret_dict['batch_size'])
        })
        return point_loss_cls, tb_dict

    def get_vote_reg_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['vote_box_labels']
        point_box_preds = self.forward_ret_dict['vote_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        xyzlwh_preds = point_box_preds[:, :6]
        xyzlwh_labels = point_box_labels[:, :6]
        point_loss_xyzlwh = self.reg_loss_func(xyzlwh_preds, xyzlwh_labels, reg_weights).sum() \
            * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][0]

        angle_bin_num = self.box_coder.angle_bin_num
        dir_cls_preds = point_box_preds[:, 6:6 + angle_bin_num]
        dir_cls_labels = point_box_labels[:, 6:6 + angle_bin_num]
        point_loss_dir_cls = F.cross_entropy(dir_cls_preds, dir_cls_labels.argmax(dim=-1), reduction='none')
        point_loss_dir_cls = (point_loss_dir_cls * reg_weights).sum() \
            * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][1]

        dir_res_preds = point_box_preds[:, 6 + angle_bin_num:6 + 2 * angle_bin_num]
        dir_res_labels = point_box_labels[:, 6 + angle_bin_num:6 + 2 * angle_bin_num]
        
        dir_res_preds = torch.sum(dir_res_preds * dir_cls_labels, dim=-1)
        dir_res_labels = torch.sum(dir_res_labels * dir_cls_labels, dim=-1)
        point_loss_dir_res = self.reg_loss_func(dir_res_preds, dir_res_labels, weights=reg_weights)
        point_loss_dir_res = point_loss_dir_res.sum() \
            * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_code_weights'][2]

        point_loss_velo = 0
        if hasattr(self.box_coder, 'pred_velo') and self.box_coder.pred_velo:
            point_loss_velo = self.reg_loss_func(
                point_box_preds[:, 6 + 2 * angle_bin_num:8 + 2 * angle_bin_num],
                point_box_labels[:, 6 + 2 * angle_bin_num:8 + 2 * angle_bin_num],
                reg_weights
            ).sum()
            tb_dict.update({
                'vote_reg_velo_loss': point_loss_velo.item()
            })

        point_loss_box = point_loss_xyzlwh + point_loss_dir_cls + point_loss_dir_res + point_loss_velo
        point_loss_box = point_loss_box * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_reg_weight']
        tb_dict.update({
            'vote_reg_loss': point_loss_box.item(),
            'vote_reg_xyzlwh_loss': point_loss_xyzlwh.item(),
            'vote_reg_dir_cls_loss': point_loss_dir_cls.item(),
            'vote_reg_dir_res_loss': point_loss_dir_res.item(),
        })
        return point_loss_box, tb_dict

    def get_vote_corner_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['gt_box_of_fg_votes']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7],
            p=self.model_cfg.LOSS_CONFIG.CORNER_LOSS_TYPE
        ).mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_corner_weight']
        tb_dict.update({'vote_corner_loss': loss_corner.item()})
        return loss_corner, tb_dict

    def voxel_knn_query_wrapper(self, points, point_coords, sp_tensor, stride, dim, query_range, radius, nsample, return_xyz=False):
        sorted_idx = torch.argsort(sp_tensor.indices[:, 0])
        new_indices = sp_tensor.indices[sorted_idx]
        new_features = sp_tensor.features[sorted_idx]
        new = SparseConvTensor(
            new_features,
            new_indices,
            sp_tensor.spatial_shape,   # same spatial dims
            sp_tensor.batch_size       # same batch size
        )
        sp_tensor = new

        coords = sp_tensor.indices
        
        voxel_xyz = common_utils.get_voxel_centers(
            coords[:, -dim:],
            downsample_times=stride,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            dim=dim
        )
        voxel_xyz = torch.cat([
            voxel_xyz,
            voxel_xyz.new_zeros((voxel_xyz.shape[0], 3-dim))
        ], dim=-1)
        v2p_ind_tensor = common_utils.generate_voxels2pinds(coords.long(), sp_tensor.spatial_shape, sp_tensor.batch_size)
        v2p_ind_tensor = v2p_ind_tensor.view(v2p_ind_tensor.shape[0], -1, v2p_ind_tensor.shape[-2], v2p_ind_tensor.shape[-1])
        stride = point_coords.new_tensor(stride)
        point_coords = torch.cat([
            point_coords[:, 0:1],
            point_coords.new_zeros((point_coords.shape[0], 3-dim)),
            torch.div(point_coords[:, -dim:], stride, rounding_mode='floor')
        ], dim=-1)
        points = torch.cat([
            points[:, :dim],
            points.new_zeros((points.shape[0], 3-dim))
        ], dim=-1)
        dist, idx, empty = voxel_knn_query(
            query_range,
            radius,
            nsample,
            voxel_xyz,
            points.contiguous(),
            point_coords.int().contiguous(),
            v2p_ind_tensor
        )
        if return_xyz:
            return dist, idx, empty, voxel_xyz, sp_tensor.features, points
        else:
            return dist, idx, empty

    def get_point_features(self, batch_dict, points, point_coords):
        point_features = []
        for k, src_name in enumerate(self.point_knn_cfg.FEATURES_SOURCE):
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            dim = 2 if src_name == 'x_bev' else 3
            layer_cfg = self.point_knn_cfg.POOL_LAYERS
            dist, idx, empty = self.voxel_knn_query_wrapper(
                points, point_coords, cur_sp_tensors, cur_stride, dim,
                layer_cfg[src_name].QUERY_RANGE, layer_cfg[src_name].RADIUS, layer_cfg[src_name].NSAMPLE
            )
            dist_recip = (1.0 / (dist + 1e-8)) * (empty == 0)
            norm = torch.sum(dist_recip, dim=-1, keepdim=True)
            weight = dist_recip / torch.clamp_min(norm, min=1e-8)
            point_features.append(k_interpolate(cur_sp_tensors.features, idx, weight))  # (N, C)
        point_features = torch.cat(point_features, dim=-1)
        point_features = self.point_feat_reduction(point_features)
        return point_features

    def get_rv_map(self, batch_dict, points, has_rv_map=True):
        batch_size = batch_dict['batch_size']
        mode = 'train' if self.training else 'test'
        rv_npoints_per_pixel = self.model_cfg.RV_CONFIG.NPOINTS_PER_PIXEL[mode]
        rv_grid_size = self.model_cfg.RV_CONFIG.GRID_SIZE
        rv_stride = self.model_cfg.RV_CONFIG.STRIDE
        azi_range = self.model_cfg.RV_CONFIG.AZI_RANGE
        dataset = self.model_cfg.RV_CONFIG.DATASET
        version = self.model_cfg.RV_CONFIG.get('VERSION', 1)
        rv_coords = generate_rv_coords(points, batch_size, rv_grid_size, azi_range, dataset, \
            batch_dict.get('transformation_3d_list', None), batch_dict.get('transformation_3d_params', None),\
            batch_dict.get('extrinsic', None), batch_dict.get('inclination', None))
        stride_rv_coords = torch.stack([
            rv_coords[:, 0],
            torch.div(rv_coords[:, 1], rv_stride[0], rounding_mode='floor'),
            torch.div(rv_coords[:, 2], rv_stride[1], rounding_mode='floor'),
        ], dim=-1)
        if not has_rv_map:
            if version == 1:
                stride_rv_map = rv_ops_utils.rv_assigner(
                    stride_rv_coords.int(), 
                    (rv_grid_size[0] // rv_stride[0], rv_grid_size[1] // rv_stride[1]),
                    rv_npoints_per_pixel * rv_stride[0] * rv_stride[1]
                )
                return stride_rv_map, stride_rv_coords, rv_coords
            else:
                xyz, ori_indices, rv_ends = rv_ops_utils.rv_assigner_v2(
                    points[:, 1:4].contiguous(),
                    stride_rv_coords.long(), 
                    (rv_grid_size[0] // rv_stride[0], rv_grid_size[1] // rv_stride[1])
                )
                return xyz, ori_indices, rv_ends
        else:
            return stride_rv_coords, rv_coords

    def get_point_indices(self, xyz, query_xyz, query_rv_coords, rv_map=None, sorted_xyz=None, ori_indices=None, rv_ends=None, xyz_coords=None):
        vote_query_cfg = self.model_cfg.VOTE_QUERY_CONFIG
        version = self.model_cfg.RV_CONFIG.get('VERSION', 1)
        if version == 1:
            idx, empty = rv_ops_utils.rv_fps_query(
                vote_query_cfg.RADIUS,
                vote_query_cfg.MAX_NSAMPLE,
                vote_query_cfg.NSAMPLE,
                vote_query_cfg.DILATION,
                vote_query_cfg.QUERY_RANGE,
                xyz.contiguous(),
                query_xyz.contiguous(),
                query_rv_coords.int().contiguous(),
                rv_map,
                'rv_rand'
            )
        elif version == 2:
            rv_grid_size = self.model_cfg.RV_CONFIG.GRID_SIZE
            rv_stride = self.model_cfg.RV_CONFIG.STRIDE
            idx, empty = rv_ops_utils.rv_fps_query_v2(
                vote_query_cfg.RADIUS,
                vote_query_cfg.MAX_NSAMPLE,
                vote_query_cfg.NSAMPLE,
                vote_query_cfg.DILATION,
                vote_query_cfg.QUERY_RANGE,
                (rv_grid_size[0] // rv_stride[0], rv_grid_size[1] // rv_stride[1]),
                sorted_xyz.contiguous(),
                query_xyz.contiguous(),
                query_rv_coords.int().contiguous(),
                ori_indices,
                rv_ends.int()
            )
        else:
            raise NotImplementedError
        xyz_mask = idx.new_zeros((xyz.shape[0],)).scatter_(0, idx.view(-1).long(), 1)
        xyz_mask_idx = torch.nonzero(xyz_mask, as_tuple=True)[0]
        xyz_mask.scatter_(0, xyz_mask_idx, torch.arange(len(xyz_mask_idx), device=xyz_mask_idx.device, dtype=xyz_mask.dtype))
        new_idx = xyz_mask[idx.long()]
        return xyz_mask_idx, new_idx, empty

    def get_bev_features(self, points, bev_features, bev_stride):
        """
        Args:
            points: (B, K, 3)
        """
        point_cloud_range = torch.tensor(self.point_cloud_range, device=points.device, dtype=torch.float32)
        voxel_size = torch.tensor(self.voxel_size, device=points.device, dtype=torch.float32)
        xy = (points[..., 0:2] - point_cloud_range[0:2]) / voxel_size[0:2] / bev_stride  # (B, K, 2)
        h, w = bev_features.shape[-2:]
        norm_xy = torch.cat([
            xy[..., 0:1] / (w - 1),
            xy[..., 1:2] / (h - 1)
        ], dim=-1)
        bev_feats = torch.nn.functional.grid_sample(
            bev_features,  # (B, C, H, W)
            norm_xy.unsqueeze(2) * 2 - 1,  # (B, K, 1, 2)
            align_corners=True
        ).squeeze(-1).permute(0, 2, 1)  # (B, C, K) -> (B, K, C)

        return bev_feats

    def forward(self, batch_dict):

        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        sf = batch_dict['spatial_features']   # the original SparseConvTensor
        
        # to avoid passing empty vectors into the model.... it panics...
        if sf.features.shape[0] == 0:
            batch_dict['batch_cls_preds']   = torch.empty((0, 1), device=sf.features.device)
            batch_dict['batch_box_preds']   = torch.empty((0, 7), device=sf.features.device)  # adjust shape to your box dim
            batch_dict['batch_index']       = torch.empty((0,),   device=sf.features.device, dtype=torch.long)
            batch_dict['cls_preds_normalized'] = False
            return batch_dict
        else:
            batch_dict['spatial_features'] = self.exp(sf)

        spatial_features = batch_dict['spatial_features']
        sorted_idx = torch.argsort(spatial_features.indices[:, 0])
        new_indices = spatial_features.indices[sorted_idx]
        new_features = spatial_features.features[sorted_idx]
        new = SparseConvTensor(
            new_features,
            new_indices,
            spatial_features.spatial_shape,   # same spatial dims
            spatial_features.batch_size       # same batch size
        )
        batch_dict['spatial_features'] = new
        spatial_features = new
        voxel_coords = spatial_features.indices
        voxel_features = spatial_features.features
        voxel_stride = batch_dict['spatial_features_stride']
        voxel_centers = common_utils.get_voxel_centers(voxel_coords[:, 1:], voxel_stride, self.voxel_size, self.point_cloud_range, dim=2)
        voxel_centers = torch.cat([
            voxel_coords[:, 0:1],
            voxel_centers,
            voxel_centers.new_full((voxel_centers.shape[0], 1), self.model_cfg.ANCHOR_HEIGHT)
        ], dim=-1)  # (N, 4), [bs_idx, x, y, z]

        batch_size = batch_dict['batch_size']
        mode = 'train' if self.training else 'test'
        seeds = voxel_centers  # (N, 4)
        seed_features = voxel_features  # (N, C)

        votes, seed_cls, seed_reg = self.vote_layer(seeds, seed_features)
        fps_indices = fps_pool_layer(
            seeds, seed_features, torch.max(seed_cls, dim=-1)[0], batch_size,
            self.model_cfg.FPS_CONFIG, mode
        )

        vote_candidates = votes[fps_indices].detach()  # (K1+K2+..., 4)
        voxel_features_dense = batch_dict['multi_scale_3d_features']['x_bev'].dense()
        voxel_features_dense = self.dense_conv2d(voxel_features_dense) + voxel_features_dense
        vote_candidate_features = self.get_bev_features(
            vote_candidates[..., 1:4].reshape(batch_size, -1, 3), 
            voxel_features_dense,
            batch_dict['multi_scale_3d_strides']['x_bev']
        )
        vote_candidate_features = vote_candidate_features.reshape(-1, vote_candidate_features.shape[-1])  # (K1+K2+..., C)
        vote_candidate_features = self.vote_reduce_conv(torch.cat([vote_candidate_features, seed_features[fps_indices]], dim=-1))
        pc_range = vote_candidates.new_tensor(self.point_cloud_range)
        voxel_size = vote_candidates.new_tensor(self.voxel_size)
        vote_candidate_coords = ((vote_candidates[:, 1:4] - pc_range[:3]) / voxel_size).to(torch.int64)
        vote_candidate_coords = torch.cat([vote_candidates[:, 0:1].long(), torch.flip(vote_candidate_coords, dims=[-1])], dim=-1)  # [bs_idx, Z, Y, X]
        _, voxel_idx, voxel_empty, voxel_k_pos, voxel_k_feats, voxe_q_pos = self.voxel_knn_query_wrapper(vote_candidates[:, 1:4], vote_candidate_coords, \
            batch_dict['multi_scale_3d_features']['x_bev'], batch_dict['multi_scale_3d_strides']['x_bev'], \
            2, [0, 16, 16], 100.0, 128, return_xyz=True
        )
        voxel_key_features = voxel_k_feats[voxel_idx.long()] \
            * (voxel_empty == 0).unsqueeze(-1)  # (K1+K2+..., T, C)
        voxel_key_pos_emb = (voxel_k_pos[voxel_idx.long()] - voxe_q_pos.unsqueeze(1)) \
            * (voxel_empty == 0).unsqueeze(-1)  # (K1+K2+..., T, 3)

        points = batch_dict['voxel_features']
        point_coords = batch_dict['voxel_coords']
        version = self.model_cfg.RV_CONFIG.get('VERSION', 1)
        if version == 1:
            stride_rv_coords, _ = self.get_rv_map(batch_dict, vote_candidates)
            stride_rv_map, _, _ = self.get_rv_map(batch_dict, torch.cat([point_coords[:, 0:1], points[:, 0:3]], dim=-1), has_rv_map=False)  # [bs_idx, x, y, z]
            pts_idx, pooled_idx, pooled_empty = self.get_point_indices(
                points[:, :3].contiguous(), vote_candidates[:, 1:4].contiguous(), \
                stride_rv_coords, stride_rv_map
            )
        elif version == 2:
            stride_rv_coords, _ = self.get_rv_map(batch_dict, vote_candidates)
            sorted_xyz, ori_indices, rv_ends = self.get_rv_map(batch_dict, torch.cat([point_coords[:, 0:1], points[:, 0:3]], dim=-1), has_rv_map=False)
            pts_idx, pooled_idx, pooled_empty = self.get_point_indices(
                points[:, :3].contiguous(), vote_candidates[:, 1:4].contiguous(), \
                stride_rv_coords, sorted_xyz=sorted_xyz, ori_indices=ori_indices, rv_ends=rv_ends
            )
        else:
            pts_idx, pooled_idx, pooled_empty = self.get_point_indices(
                points[:, :3].contiguous(), vote_candidates[:, 1:4].contiguous(), \
                vote_candidates[:, 0:1], xyz_coords=point_coords[:, 0:1]
            )

        new_points = points[pts_idx]
        new_point_coords = point_coords[pts_idx]
        new_point_features = self.get_point_features(batch_dict, new_points[:, :3], new_point_coords)

        key_features = new_point_features[pooled_idx.long()] \
            * (pooled_empty == 0).unsqueeze(-1).unsqueeze(-1)  # (K1+K2+..., T, C)
        key_pos_emb = (new_points[:, :3][pooled_idx.long()] - vote_candidates[:, 1:4].unsqueeze(1)) \
            * (pooled_empty == 0).unsqueeze(-1).unsqueeze(-1)  # (K1+K2+..., T, 3)

        vote_features = self.pv_transformer(
            src=torch.cat([key_features.permute(1, 0, 2), voxel_key_features.permute(1, 0, 2)], dim=0),
            tgt=vote_candidate_features.unsqueeze(1).permute(1, 0, 2),
            pos_res=torch.cat([key_pos_emb.permute(1, 0, 2).unsqueeze(0), voxel_key_pos_emb.permute(1, 0, 2).unsqueeze(0)], dim=1)
        ).squeeze(0)

        vote_features = self.shared_conv(vote_features)
        vote_cls_preds = self.cls_conv(vote_features)
        vote_box_preds = self.box_conv(vote_features)

        ret_dict = {
            'vote_cls_preds': vote_cls_preds,
            'vote_box_preds': vote_box_preds,
            'votes': vote_candidates,
            'votes_list': [votes],
            'seeds_list': [seeds],
            'seeds_cls_list': [seed_cls],
            'batch_size': batch_dict['batch_size']
        }

        batch_dict.update({
            'votes_list': ret_dict['votes_list'],
            'seeds_list': ret_dict['seeds_list'],
            'votes': ret_dict['votes']
        })
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training or \
                self.model_cfg.LOSS_CONFIG.PREDICT_BOXES:

            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=ret_dict['votes'][:, 1:4],
                point_cls_preds=vote_cls_preds, point_box_preds=vote_box_preds
            )

            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = ret_dict['votes'][:, 0].contiguous()
            batch_dict['cls_preds_normalized'] = False

            ret_dict['point_box_preds'] = point_box_preds

        self.forward_ret_dict = ret_dict

        return batch_dict
