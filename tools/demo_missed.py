import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import pickle
from pathlib import Path

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None,args=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        


        # e.g. cfg.DATA_CONFIG.DATA_PATH = 'data/kitti'
        # and your infos file is in that folder
        # split = dataset_cfg.DATA_SPLIT  # usually 'training'

        if args is not None:
            info_file = Path(args.info_path)
        
        with open(info_file, 'rb') as f:
            infos = pickle.load(f)

        # map lidar index (e.g. 0,1,2,...) → the annos dict
        # each info has info['point_cloud']['lidar_idx'] and info['annos']
        self.info_dict = {
            info['point_cloud']['lidar_idx']: info['annos']
            for info in infos
        }
        # print(self.info_dict.keys())

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        stem  = Path(self.sample_file_list[index]).stem
        annos = self.info_dict.get(stem, None)

        if annos is not None:
            # these are already in LiDAR coords: (N,7)
            input_dict['gt_boxes'] = annos['gt_boxes_lidar']

            # optional: names if you want color/class tags
            input_dict['gt_names'] = annos['name']
        data_dict = self.prepare_data(data_dict=input_dict)
        # print(data_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--info_path', type=str, help='path to your kitti infos .pkl')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger,args=args
    )



    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset, logger=logger)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_boxes  = pred_dicts[0]['pred_boxes'].cpu().numpy()   # (M,7)
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()  # (M,)
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()  # (M,)
            # print(pred_boxes.shape)

            # get GT boxes from dataset (if your DatasetTemplate provides 'gt_boxes')
            # shape will be (1, N, 7) after collate; take [0]
            if('gt_boxes' not in data_dict):
                print()
                print('----------------------------------------')
                print("No ground truth labels for this image")
                print('----------------------------------------')
                exit(0)
            gt_boxes = data_dict['gt_boxes'][0]
            gt_boxes = gt_boxes.cpu().numpy()         # (N,8)
            gt_labels = gt_boxes[:, 7].astype(int)
            gt_boxes  = gt_boxes[:, :7]

            gt_t = torch.from_numpy(gt_boxes).float().cuda().contiguous()
            pd_t = torch.from_numpy(pred_boxes).float().cuda().contiguous()
            # if(pred_boxes.shape[0] == 0) :
            #     print("No predicted box for this!!")
            #     print("Will exit")
            #     exit(0)
            # compute IoU matrix (N_gt × N_pred)
            # iou_mat = None
            if gt_boxes.shape[0] == 0:
            # no GT: nothing to miss
                missed_gt_boxes = np.zeros((0,7), dtype=float)
            elif pred_boxes.shape[0] == 0:
                # no preds: all GT are missed
                missed_gt_boxes = gt_boxes.copy()
            else:
                # normal IoU path
                for g, p in zip(gt_boxes, pred_boxes):
                    print('Δcenter:', np.round(p[:3] - g[:3], 3),
                        'Δsize:',   np.round(p[3:6] - g[3:6], 3),
                        'Δyaw:',    np.round(p[6]  - g[6],      3))
                    
                gt_t = torch.from_numpy(gt_boxes).float().cuda().contiguous()
                pd_t = torch.from_numpy(pred_boxes).float().cuda().contiguous()
                iou_mat = boxes_iou3d_gpu(gt_t, pd_t).cpu().numpy()   # (N_gt, N_pred)
                max_iou_per_gt = iou_mat.max(axis=1)                  # shape (N_gt,)
                IOU_THRESHOLD    = 0.5
                missed_mask      = max_iou_per_gt < IOU_THRESHOLD
                missed_gt_boxes  = gt_boxes.copy()
            # max_iou_per_gt = iou_mat.max(axis=1)
            # (K,7)

            # draw everything:
            #  - predicted boxes (green)
            #  - missed GT boxes (orange)
            if OPEN3D_FLAG:
            #     #draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
                V.draw_scenes(
                    points     = data_dict['points'][:, 1:],
                    ref_boxes  = pred_boxes,
                    ref_scores = pred_scores,
                    ref_labels = pred_labels,
                    gt_boxes   = missed_gt_boxes,
                )
            else:
                # mayavi version: draw predictions first (green)
                # V.draw_scenes(
                #     points     = data_dict['points'][:, 1:],
                #     ref_boxes  = pred_boxes,
                #     ref_scores = pred_scores,
                #     ref_labels = pred_labels
                # )
                # then overlay missed GT
                V.draw_corners3d(
                    V.boxes_to_corners3d(missed_gt_boxes), 
                    color=(1.0, 0.5, 0.0),    # orange
                    fig=mlab.gcf()
                )
                mlab.show(stop=True)

            if not OPEN3D_FLAG:
                pass
                # mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
