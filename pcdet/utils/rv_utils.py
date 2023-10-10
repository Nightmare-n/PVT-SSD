
import torch
from torch.nn import functional as F
from . import transform_utils
from ..ops.rv_ops import rv_ops_utils
import numpy as np


@torch.no_grad()
def generate_points2rvinds(points, rv_grid_size, azi_range, extrinsic=None, inclination=None, dataset='KITTI'):
    h, w  = rv_grid_size
    azi_range = torch.tensor(azi_range, dtype=torch.float32).to(points.device) * np.pi
    if dataset == 'KITTI':
        height = torch.tensor([
            0.20966667, 0.2092    , 0.2078    , 0.2078    , 0.2078    ,
            0.20733333, 0.20593333, 0.20546667, 0.20593333, 0.20546667,
            0.20453333, 0.205     , 0.2036    , 0.20406667, 0.2036    ,
            0.20313333, 0.20266667, 0.20266667, 0.20173333, 0.2008    ,
            0.2008    , 0.2008    , 0.20033333, 0.1994    , 0.20033333,
            0.19986667, 0.1994    , 0.1994    , 0.19893333, 0.19846667,
            0.19846667, 0.19846667, 0.12566667, 0.1252    , 0.1252    ,
            0.12473333, 0.12473333, 0.1238    , 0.12333333, 0.1238    ,
            0.12286667, 0.1224    , 0.12286667, 0.12146667, 0.12146667,
            0.121     , 0.12053333, 0.12053333
        ], dtype=torch.float32).to(points.device)
        zenith = torch.tensor([
            0.03373091,  0.02740409,  0.02276443,  0.01517224,  0.01004049,
            0.00308099, -0.00155868, -0.00788549, -0.01407172, -0.02103122,
            -0.02609267, -0.032068  , -0.03853542, -0.04451074, -0.05020488,
            -0.0565317 , -0.06180405, -0.06876355, -0.07361411, -0.08008152,
            -0.08577566, -0.09168069, -0.09793721, -0.10398284, -0.11052055,
            -0.11656618, -0.12219002, -0.12725147, -0.13407038, -0.14067839,
            -0.14510716, -0.15213696, -0.1575499 , -0.16711043, -0.17568678,
            -0.18278688, -0.19129293, -0.20247031, -0.21146846, -0.21934183,
            -0.22763699, -0.23536977, -0.24528179, -0.25477201, -0.26510582,
            -0.27326038, -0.28232882, -0.28893683
        ], dtype=torch.float32).to(points.device)
        incl = -zenith
        xy_norm = torch.linalg.norm(points[:, :2], ord=2, dim=-1)
        error = torch.abs(incl.unsqueeze(0) - torch.atan2(height.unsqueeze(0) - points[:, 2:3], xy_norm.unsqueeze(1)))
        row_inds = torch.argmin(error, dim=-1)
        azi = torch.atan2(points[:, 1], points[:, 0])
        col_inds = w - 1.0 + 0.5 - (azi - azi_range[0]) / (azi_range[1] - azi_range[0]) * w
        col_inds = torch.round(col_inds).long()
        rv_coords = torch.stack([row_inds, col_inds], dim=-1)
    elif dataset == 'WAYMO':
        extrinsic = extrinsic.to(points.dtype)
        inclination = inclination.to(points.dtype)
        vehicle_to_laser = torch.linalg.inv(extrinsic)
        rotation = vehicle_to_laser[:3, :3]  # (3, 3)
        translation = vehicle_to_laser[:3, 3].unsqueeze(0)  # (1, 3)
        points = torch.einsum('ij,kj->ik', points, rotation) + translation
        xy_norm = torch.linalg.norm(points[:, :2], ord=2, dim=-1)
        error = torch.abs(inclination.unsqueeze(0) - torch.atan2(points[:, 2:3], xy_norm.unsqueeze(1)))
        row_inds = torch.argmin(error, dim=-1)
        az_correction = torch.atan2(extrinsic[1, 0], extrinsic[0, 0])
        point_azimuth = torch.atan2(points[:, 1], points[:, 0]) + az_correction
        point_azimuth_gt_pi_mask = point_azimuth > np.pi
        point_azimuth_lt_minus_pi_mask = point_azimuth < -np.pi
        point_azimuth = point_azimuth - point_azimuth_gt_pi_mask.to(points.dtype) * 2 * np.pi
        point_azimuth = point_azimuth + point_azimuth_lt_minus_pi_mask.to(points.dtype) * 2 * np.pi
        col_inds = w - 1.0 + 0.5 - (point_azimuth - azi_range[0]) / (azi_range[1] - azi_range[0]) * w
        col_inds = torch.round(col_inds).long()
        rv_coords = torch.stack([row_inds, col_inds], dim=-1)
    else:
        raise NotImplementedError
    return rv_coords


@torch.no_grad()
def generate_rv_coords(points, batch_size, rv_grid_size, azi_range, dataset,
                       transformation_3d_list=None, transformation_3d_params=None,
                       batch_extrinsic=None, batch_inclination=None):
    rv_coords = []
    for bs_idx in range(batch_size):
        cur_xyz = points[points[:, 0] == bs_idx][:, 1:4].clone()
        if transformation_3d_list is not None:
            cur_3d_trans_list = transformation_3d_list[bs_idx]
            cur_3d_trans_params = transformation_3d_params[bs_idx]
            for key in cur_3d_trans_list[::-1]:
                cur_xyz, _ = getattr(transform_utils, key)(cur_3d_trans_params[key], reverse=True, points_3d=cur_xyz)
        extrinsic = batch_extrinsic[bs_idx] if batch_extrinsic is not None else None
        inclination = batch_inclination[bs_idx] if batch_inclination is not None else None
        cur_rv_coords = generate_points2rvinds(cur_xyz, rv_grid_size, azi_range, \
            extrinsic=extrinsic, inclination=inclination, dataset=dataset)
        rv_coords.append(F.pad(cur_rv_coords, (1, 0), mode='constant', value=bs_idx))
    rv_coords = torch.cat(rv_coords, dim=0)
    return rv_coords


@torch.no_grad()
def generate_rv_map(rv_coords, rv_stride, rv_grid_size, rv_npoints_per_pixel):
    stride_rv_coords = torch.stack([
        rv_coords[:, 0],
        torch.div(rv_coords[:, 1], rv_stride[0], rounding_mode='floor'),
        torch.div(rv_coords[:, 2], rv_stride[1], rounding_mode='floor'),
    ], dim=-1)
    stride_rv_map = rv_ops_utils.rv_assigner(
        stride_rv_coords.int(), 
        (rv_grid_size[0] // rv_stride[0], rv_grid_size[1] // rv_stride[1]),
        rv_npoints_per_pixel * rv_stride[0] * rv_stride[1]
    )
    return stride_rv_map, stride_rv_coords
