from scipy.linalg import orthogonal_procrustes
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import math
import random
from glob import glob
from pycocotools.coco import COCO
from main.config import cfg
from common.utils.human_models import mano
from common.utils.preprocessing import load_img, get_bbox, process_bbox, augmentation, get_iou, load_ply
from common.utils.vis import vis_keypoints, save_obj


def compute_rigid_transform(pred_points, gt_points):
    """
    Computes rotation and translation using Procrustes analysis.
    pred_points: (N, 3)
    gt_points: (N, 3)
    returns: R (3,3), t (3,)
    """
    assert pred_points.shape == gt_points.shape
    pred_center = pred_points.mean(axis=0)
    gt_center = gt_points.mean(axis=0)

    pred_centered = pred_points - pred_center
    gt_centered = gt_points - gt_center

    # Solve orthogonal Procrustes problem
    R, _ = orthogonal_procrustes(pred_centered, gt_centered)

    # Compute translation
    t = gt_center - R.dot(pred_center)

    return R, t


class HIC(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        assert data_split == 'test', 'only testing is supported for HIC dataset'
        self.data_path = osp.join('HIC', 'data')
        self.focal = (525.0, 525.0)
        self.princpt = (319.5, 239.5)

        # HIC joint set
        self.joint_set = {
            'joint_num': 28,
            'joints_name': (
            'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Middle_1', 'R_Middle_2',
            'R_Middle_3', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Thumb_2', 'R_Thumb_3', 'L_Pinky_1', 'L_Pinky_2',
            'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Index_3',
            'L_Index_2', 'L_Index_1', 'L_Thumb_2', 'L_Thumb_3'),
            'flip_pairs': [(i, i + 14) for i in range(14)]
        }
        self.joint_set['joint_type'] = {'right': np.arange(0, self.joint_set['joint_num'] // 2),
                                        'left': np.arange(self.joint_set['joint_num'] // 2,
                                                          self.joint_set['joint_num'])}
        self.datalist = self.load_data()

    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.data_path, 'HIC.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.data_path, img['file_name'])
            hand_type = ann['hand_type']

            # bbox
            body_bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
            body_bbox = process_bbox(body_bbox, img_width, img_height, extend_ratio=1.0)
            if body_bbox is None:
                continue

            # mano mesh
            if ann['right_mano_path'] is not None:
                right_mano_path = osp.join(self.data_path, ann['right_mano_path'])
            else:
                right_mano_path = None
            if ann['left_mano_path'] is not None:
                left_mano_path = osp.join(self.data_path, ann['left_mano_path'])
            else:
                left_mano_path = None

            datalist.append({
                'aid': aid,
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'body_bbox': body_bbox,
                'hand_type': hand_type,
                'right_mano_path': right_mano_path,
                'left_mano_path': left_mano_path})

        return datalist

    def get_bbox_from_mesh(self, mesh):
        x = mesh[:, 0] / mesh[:, 2] * self.focal[0] + self.princpt[0]
        y = mesh[:, 1] / mesh[:, 2] * self.focal[1] + self.princpt[1]
        xy = np.stack((x, y), 1)
        bbox = get_bbox(xy, np.ones_like(x))
        return bbox

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, body_bbox = data['img_path'], data['img_shape'], data['body_bbox']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, body_bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        # mano coordinates
        right_mano_path = data['right_mano_path']
        if right_mano_path is not None:
            rhand_mesh = load_ply(right_mano_path)
        else:
            rhand_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
        left_mano_path = data['left_mano_path']
        if left_mano_path is not None:
            lhand_mesh = load_ply(left_mano_path)
        else:
            lhand_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
        mano_mesh_cam = np.concatenate((rhand_mesh, lhand_mesh))

        inputs = {'img': img}
        targets = {'mano_mesh_cam': mano_mesh_cam}
        meta_info = {'bb2img_trans': bb2img_trans}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
            'mpvpe_sh': [None for _ in range(sample_num)],
            'mpvpe_ih': [None for _ in range(sample_num * 2)],
            'mpjpe_sh': [None for _ in range(sample_num)],
            'mpjpe_ih': [None for _ in range(sample_num * 2)],
            'rrve': [None for _ in range(sample_num)],
            'mrrpe': [None for _ in range(sample_num)],
            'pa_mpjpe_sh': [None] * sample_num,
            'pa_mpjpe_ih': [None] * (sample_num * 2),
            'pa_mpvpe_sh': [None] * sample_num,
            'pa_mpvpe_ih': [None] * (sample_num * 2),
            'f5_sh': [None] * sample_num,
            'f5_ih': [None] * (sample_num * 2),
            'f10_sh': [None] * sample_num,
            'f10_ih': [None] * (sample_num * 2),
            'f15_sh': [None] * sample_num,
            'f15_ih': [None] * (sample_num * 2)
        }

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]

            out = outs[n]
            mesh_out = np.concatenate((out['rmano_mesh_cam'], out['lmano_mesh_cam'])) * 1000  # meter to milimeter
            mesh_gt = out['mano_mesh_cam_target'] * 1000  # meter to milimeter

            # visualize
            vis = False
            if vis:
                filename = str(annot['aid'])
                img = out['img'].transpose(1, 2, 0)[:, :, ::-1] * 255

                lhand_bbox = out['lhand_bbox'].reshape(2, 2).copy()
                lhand_bbox[:, 0] = lhand_bbox[:, 0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                lhand_bbox[:, 1] = lhand_bbox[:, 1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                lhand_bbox = lhand_bbox.reshape(4)
                img = cv2.rectangle(img.copy(), (int(lhand_bbox[0]), int(lhand_bbox[1])),
                                    (int(lhand_bbox[2]), int(lhand_bbox[3])), (255, 0, 0), 3)
                rhand_bbox = out['rhand_bbox'].reshape(2, 2).copy()
                rhand_bbox[:, 0] = rhand_bbox[:, 0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                rhand_bbox[:, 1] = rhand_bbox[:, 1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                rhand_bbox = rhand_bbox.reshape(4)
                img = cv2.rectangle(img.copy(), (int(rhand_bbox[0]), int(rhand_bbox[1])),
                                    (int(rhand_bbox[2]), int(rhand_bbox[3])), (0, 0, 255), 3)
                cv2.imwrite(filename + '.jpg', img)

                save_obj(out['rmano_mesh_cam'], mano.face['right'], filename + '_right.obj')
                save_obj(out['lmano_mesh_cam'] + out['rel_trans'].reshape(1, 3), mano.face['left'],
                         filename + '_left.obj')

            # mrrpe
            rel_trans_gt = np.dot(mano.sh_joint_regressor, mesh_gt[mano.vertex_num:, :])[mano.sh_root_joint_idx] - \
                           np.dot(mano.sh_joint_regressor, mesh_gt[:mano.vertex_num, :])[mano.sh_root_joint_idx]
            rel_trans_out = out['rel_trans'] * 1000  # meter to milimeter
            if annot['hand_type'] == 'interacting':
                eval_result['mrrpe'][n] = np.sqrt(np.sum((rel_trans_gt - rel_trans_out) ** 2))

            # root joint alignment
            for h in ('right', 'left'):
                if h == 'right':
                    vertex_mask = np.arange(0, mano.vertex_num)
                else:
                    vertex_mask = np.arange(mano.vertex_num, 2 * mano.vertex_num)
                mesh_gt[vertex_mask, :] = mesh_gt[vertex_mask, :] - np.dot(mano.sh_joint_regressor,
                                                                           mesh_gt[vertex_mask, :])[
                                                                    mano.sh_root_joint_idx, None, :]
                mesh_out[vertex_mask, :] = mesh_out[vertex_mask, :] - np.dot(mano.sh_joint_regressor,
                                                                             mesh_out[vertex_mask, :])[
                                                                      mano.sh_root_joint_idx, None, :]

            # mpvpe
            if annot['hand_type'] == 'right' and annot['right_mano_path'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(
                    np.sum((mesh_gt[:mano.vertex_num, :] - mesh_out[:mano.vertex_num, :]) ** 2, 1)).mean()
            elif annot['hand_type'] == 'left' and annot['left_mano_path'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(
                    np.sum((mesh_gt[mano.vertex_num:, :] - mesh_out[mano.vertex_num:, :]) ** 2, 1)).mean()
            elif annot['hand_type'] == 'interacting':
                if annot['right_mano_path'] is not None:
                    eval_result['mpvpe_ih'][2 * n] = np.sqrt(
                        np.sum((mesh_gt[:mano.vertex_num, :] - mesh_out[:mano.vertex_num, :]) ** 2, 1)).mean()
                if annot['left_mano_path'] is not None:
                    eval_result['mpvpe_ih'][2 * n + 1] = np.sqrt(
                        np.sum((mesh_gt[mano.vertex_num:, :] - mesh_out[mano.vertex_num:, :]) ** 2, 1)).mean()

            # mpjpe_sh
            joint_gt_sh_right = np.dot(mano.sh_joint_regressor, mesh_gt[:mano.vertex_num, :])
            joint_gt_sh_left = np.dot(mano.sh_joint_regressor, mesh_gt[mano.vertex_num:, :])
            joint_out_sh_right = np.dot(mano.sh_joint_regressor, mesh_out[:mano.vertex_num, :])
            joint_out_sh_left = np.dot(mano.sh_joint_regressor, mesh_out[mano.vertex_num:, :])

            # PA
            def process_pa(joint_out, joint_gt, mesh_out_part, mesh_gt_part):
                if joint_out.shape[0] < 3 or joint_gt.shape[0] < 3:
                    return None, None, None, None
                try:
                    R, t = compute_rigid_transform(joint_out, joint_gt)
                    aligned_joint = (joint_out @ R.T) + t
                    pa_mpjpe = np.sqrt(((aligned_joint - joint_gt) ** 2).sum(1)).mean()

                    aligned_mesh = (mesh_out_part @ R.T) + t
                    pa_mpvpe = np.sqrt(((aligned_mesh - mesh_gt_part) ** 2).sum(1)).mean()

                    vertex_errors = np.sqrt(((aligned_mesh - mesh_gt_part) ** 2).sum(1))
                    f_score_5mm = (vertex_errors < 5.0).mean()
                    f_score_10mm = (vertex_errors < 10.0).mean()
                    f_score_15mm = (vertex_errors < 15.0).mean()
                    return pa_mpjpe, pa_mpvpe, f_score_5mm, f_score_10mm, f_score_15mm, vertex_errors
                except:
                    return None, None, None, None

            if annot['hand_type'] == 'right' and annot['right_mano_path'] is not None:
                # PA
                pa_mpjpe, pa_mpvpe, f5, f10, f15, _  = process_pa(
                    joint_out_sh_right, joint_gt_sh_right,
                    mesh_out[:mano.vertex_num], mesh_gt[:mano.vertex_num]
                )
                if pa_mpjpe is not None:
                    eval_result['pa_mpjpe_sh'][n] = pa_mpjpe
                    eval_result['pa_mpvpe_sh'][n] = pa_mpvpe
                    eval_result['f5_sh'][n] = f5
                    eval_result['f10_sh'][n] = f10
                    eval_result['f15_sh'][n] = f15
                eval_result['mpjpe_sh'][n] = np.sqrt(
                    np.sum((joint_gt_sh_right - joint_out_sh_right) ** 2, 1)).mean()
            elif annot['hand_type'] == 'left' and annot['left_mano_path'] is not None:
                pa_mpjpe, pa_mpvpe, f5, f10, f15, _ = process_pa(
                    joint_out_sh_left, joint_gt_sh_left,
                    mesh_out[mano.vertex_num:], mesh_gt[mano.vertex_num:]
                )
                if pa_mpjpe is not None:
                    eval_result['pa_mpjpe_sh'][n] = pa_mpjpe
                    eval_result['pa_mpvpe_sh'][n] = pa_mpvpe
                    eval_result['f5_sh'][n] = f5
                    eval_result['f10_sh'][n] = f10
                    eval_result['f15_sh'][n] = f15
                eval_result['mpjpe_sh'][n] = np.sqrt(
                    np.sum((joint_gt_sh_left - joint_out_sh_left) ** 2, 1)).mean()
            elif annot['hand_type'] == 'interacting':
                idx=0
                if annot['right_mano_path'] is not None:
                    pa_mpjpe, pa_mpvpe, f5, f10, f15, _ = process_pa(
                        joint_out_sh_right, joint_gt_sh_right,
                        mesh_out[:mano.vertex_num], mesh_gt[:mano.vertex_num]
                    )
                    if pa_mpjpe is not None:
                        eval_result['pa_mpjpe_ih'][2 * n] = pa_mpjpe
                        eval_result['pa_mpvpe_ih'][2 * n] = pa_mpvpe
                        eval_result['f5_ih'][2 * n] = f5
                        eval_result['f10_ih'][2 * n] = f10
                        eval_result['f15_ih'][2 * n] = f15
                    eval_result['mpjpe_ih'][2 * n] = np.sqrt(
                        np.sum((joint_gt_sh_right - joint_out_sh_right) ** 2, 1)).mean()
                if annot['left_mano_path'] is not None:
                    pa_mpjpe, pa_mpvpe, f5, f10, f15, _= process_pa(
                        joint_out_sh_left, joint_gt_sh_left,
                        mesh_out[mano.vertex_num:], mesh_gt[mano.vertex_num:]
                    )
                    if pa_mpjpe is not None:
                        eval_result['pa_mpjpe_ih'][2 * n + 1] = pa_mpjpe
                        eval_result['pa_mpvpe_ih'][2 * n + 1] = pa_mpvpe
                        eval_result['f5_ih'][2 * n + 1] = f5
                        eval_result['f10_ih'][2 * n + 1] = f10
                        eval_result['f15_ih'][2 * n + 1] = f15
                    eval_result['mpjpe_ih'][2 * n + 1] = np.sqrt(
                        np.sum((joint_gt_sh_left - joint_out_sh_left) ** 2, 1)).mean()

            # rrve
            if annot['hand_type'] == 'interacting':
                if annot['right_mano_path'] is not None and annot['left_mano_path'] is not None:
                    vertex_mask = np.arange(mano.vertex_num, 2 * mano.vertex_num)
                    mesh_gt[vertex_mask, :] = mesh_gt[vertex_mask, :] + rel_trans_gt
                    mesh_out[vertex_mask, :] = mesh_out[vertex_mask, :] + rel_trans_out
                    eval_result['rrve'][n] = np.sqrt(np.sum((mesh_gt - mesh_out) ** 2, 1)).mean()

        return eval_result

    def print_eval_result(self, eval_result):
        tot_eval_result = {
            'mpvpe_sh': [],
            'mpvpe_ih': [],
            'rrve': [],
            'mrrpe': [],
            'mpjpe_sh': [],
            'mpjpe_ih': [],
            'pa_mpjpe_sh': [],
            'pa_mpjpe_ih': [],
            'pa_mpvpe_sh': [],
            'pa_mpvpe_ih': [],
            'f5_sh': [],
            'f5_ih': [],
            'f10_sh': [],
            'f10_ih': [],
            'f15_sh': [],
            'f15_ih': []
        }
        for key in ['pa_mpjpe_sh', 'pa_mpjpe_ih', 'pa_mpvpe_sh', 'pa_mpvpe_ih',
                    'f5_sh', 'f5_ih', 'f10_sh', 'f10_ih',
                    'f15_sh', 'f15_ih']:
            for val in eval_result[key]:
                if val is not None:
                    tot_eval_result[key].append(val)

        # mpvpe (average all samples)
        for mpvpe_sh in eval_result['mpvpe_sh']:
            if mpvpe_sh is not None:
                tot_eval_result['mpvpe_sh'].append(mpvpe_sh)
        for mpvpe_ih in eval_result['mpvpe_ih']:
            if mpvpe_ih is not None:
                tot_eval_result['mpvpe_ih'].append(mpvpe_ih)
        for mpvpe_ih in eval_result['rrve']:
            if mpvpe_ih is not None:
                tot_eval_result['rrve'].append(mpvpe_ih)

        # mrrpe (average all samples)
        for mrrpe in eval_result['mrrpe']:
            if mrrpe is not None:
                tot_eval_result['mrrpe'].append(mrrpe)

        # mpjpe
        for mpjpe_sh in eval_result['mpjpe_sh']:
            if mpjpe_sh is not None:
                tot_eval_result['mpjpe_sh'].append(mpjpe_sh)
        for mpjpe_ih in eval_result['mpjpe_ih']:
            if mpjpe_ih is not None:
                tot_eval_result['mpjpe_ih'].append(mpjpe_ih)

        # print evaluation results
        eval_result = tot_eval_result

        print('MRRPE: %.2f mm' % (np.mean(eval_result['mrrpe'])))
        print('RRVE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['rrve'])))
        print('MPVPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'] + eval_result['mpvpe_ih'])))
        print('MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'])))
        print('MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_ih'])))
        print('MPJPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_sh'] + eval_result['mpjpe_ih'])))
        print('MPJPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_sh'])))
        print('MPJPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_ih'])))

        print('PA-MPVPE for all hand sequences: %.2f mm' % (
            np.mean(eval_result['pa_mpvpe_sh'] + eval_result['pa_mpvpe_ih'])))
        print('PA-MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['pa_mpvpe_sh'])))
        print('PA-MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['pa_mpvpe_ih'])))
        print('PA-MPJPE for all hand sequences: %.2f mm' % (
            np.mean(eval_result['pa_mpjpe_sh'] + eval_result['pa_mpjpe_ih'])))
        print('PA-MPJPE for single hand sequences: %.2f mm' % (np.mean(eval_result['pa_mpjpe_sh'])))
        print('PA-MPJPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['pa_mpjpe_ih'])))

        print('F@5mm for all hand sequences: %.2f%%' % (
                np.mean(eval_result['f5_sh'] + eval_result['f5_ih']) * 100))
        print('F@5mm for single hand sequences: %.2f%%' % (np.mean(eval_result['f5_sh']) * 100))
        print('F@5mm for interacting hand sequences: %.2f%%' % (np.mean(eval_result['f5_ih']) * 100))
        print('F@10mm for all hand sequences: %.2f%%' % (
                np.mean(eval_result['f10_sh'] + eval_result['f10_ih']) * 100))
        print('F@10mm for single hand sequences: %.2f%%' % (np.mean(eval_result['f10_sh']) * 100))
        print('F@10mm for interacting hand sequences: %.2f%%' % (np.mean(eval_result['f10_ih']) * 100))
        print('F@15mm for all hand sequences: %.2f%%' % (
                np.mean(eval_result['f15_sh'] + eval_result['f15_ih']) * 100))
        print('F@15mm for single hand sequences: %.2f%%' % (np.mean(eval_result['f15_sh']) * 100))
        print('F@15mm for interacting hand sequences: %.2f%%' % (np.mean(eval_result['f15_ih']) * 100))

        return eval_result

