import torch
import torch.nn as nn
import torch.profiler as profile
from torch.profiler import ProfilerActivity, profile, record_function
from torch.nn import functional as F

# windows vscode
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.nets.rcvit import RCViTBackboneNet
from common.nets.timm_tinyvit import TinyViTBackbone
from common.nets.resnet import ResNetBackbone
from common.nets.module import PositionNet as PositionNet01
from common.nets.module import RotationNet as RotationNet01
from common.nets.loss import CoordLoss, ParamLoss, KLLoss, klLoss, DiceLoss, CosineSimilarityLoss, SimilarityPreservingLoss
from common.utils.human_models import mano
from common.utils.transforms import rot6d_to_axis_angle
from main.config import cfg
import time
import numpy as np
import copy

class Model(nn.Module):
    def __init__(self, hand_backbone, hand_position_net, hand_rotation_net):
        super(Model, self).__init__()
        self.hand_backbone = hand_backbone
        self.hand_position_net = hand_position_net
        self.hand_rotation_net = hand_rotation_net

        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()
        self.coord_loss = CoordLoss()
        self.coord_loss_t = CoordLoss()
        self.Loss_t = CosineSimilarityLoss()
        self.param_loss = ParamLoss()

        self.trainable_modules = [self.hand_backbone, self.hand_position_net, self.hand_rotation_net]
        self.bce = nn.BCELoss()


    def forward_rotation_net(self, rhand_feat, lhand_feat, rhand_coord, lhand_coord):
        rroot_pose_6d, rpose_param_6d, rshape_param, rcam_param, lroot_pose_6d, lpose_param_6d, lshape_param, lcam_param, rel_trans = self.hand_rotation_net(rhand_feat, lhand_feat, rhand_coord, lhand_coord)
        rroot_pose = rot6d_to_axis_angle(rroot_pose_6d).reshape(-1,3)
        rpose_param = rot6d_to_axis_angle(rpose_param_6d.view(-1,6)).reshape(-1,(mano.orig_joint_num-1)*3)
        lroot_pose = rot6d_to_axis_angle(lroot_pose_6d).reshape(-1,3)
        lpose_param = rot6d_to_axis_angle(lpose_param_6d.view(-1,6)).reshape(-1,(mano.orig_joint_num-1)*3)
        return rroot_pose, rpose_param, rshape_param, rcam_param, lroot_pose, lpose_param, lshape_param, lcam_param, rel_trans

    def get_coord(self, root_pose, hand_pose, shape, cam_param, hand_type):
        batch_size = root_pose.shape[0]
        if hand_type == 'right':
            output = self.mano_layer_right(betas=shape, hand_pose=hand_pose, global_orient=root_pose)
        else:
            output = self.mano_layer_left(betas=shape, hand_pose=hand_pose, global_orient=root_pose)

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.sh_joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)

        # project 3D coordinates to 2D space
        x = joint_cam[:,:,0] * cam_param[:,None,0] + cam_param[:,None,1]
        y = joint_cam[:,:,1] * cam_param[:,None,0] + cam_param[:,None,2]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,mano.sh_root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam, root_cam

    def forward(self, inputs, targets=None, meta_info=None, mode=None):
        # body network
        hand_feat = self.hand_backbone(inputs['img'])
        rjoint_img, ljoint_img, rhand_feat, lhand_feat = self.hand_position_net(hand_feat)
        rroot_pose, rhand_pose, rshape, rcam_param, lroot_pose, lhand_pose, lshape, lcam_param, rel_trans = self.forward_rotation_net(
            rhand_feat, lhand_feat, rjoint_img.detach(), ljoint_img.detach())

        # get outputs
        ljoint_proj, ljoint_cam, lmesh_cam, lroot_cam = self.get_coord(lroot_pose, lhand_pose, lshape, lcam_param,
                                                                       'left')
        rjoint_proj, rjoint_cam, rmesh_cam, rroot_cam = self.get_coord(rroot_pose, rhand_pose, rshape, rcam_param,
                                                                       'right')

        # combine outputs for the loss calculation (follow mano.th_joints_name)
        mano_pose = torch.cat((rroot_pose, rhand_pose, lroot_pose, lhand_pose), 1)
        mano_shape = torch.cat((rshape, lshape), 1)
        joint_cam = torch.cat((rjoint_cam, ljoint_cam), 1)
        joint_img = torch.cat((rjoint_img, ljoint_img), 1)
        joint_proj = torch.cat((rjoint_proj, ljoint_proj), 1)

        if mode == 'train':
            loss = {}
            loss['rel_trans'] = self.coord_loss(rel_trans[:, None, :], targets['rel_trans'][:, None, :],
                                                meta_info['rel_trans_valid'][:, None, :])
            loss['mano_pose'] = self.param_loss(mano_pose, targets['mano_pose'], meta_info['mano_param_valid'])
            loss['mano_shape'] = self.param_loss(mano_shape, targets['mano_shape'], meta_info['mano_shape_valid'])
            loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'],
                                                meta_info['joint_valid'] * meta_info['is_3D'][:, None, None]) * 10
            loss['mano_joint_cam'] = self.coord_loss(joint_cam, targets['mano_joint_cam'],
                                                     meta_info['mano_joint_valid']) * 10
            loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'],
                                                meta_info['is_3D'])
            loss['mano_joint_img'] = self.coord_loss(joint_img, targets['mano_joint_img'],
                                                     meta_info['mano_joint_trunc'])
            loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:, :, :2], meta_info['joint_valid'])
            return loss
        else:
            # test output
            out = {}
            out['img'] = inputs['img']
            out['rel_trans'] = rel_trans
            out['rjoint_img'] = rjoint_img
            out['ljoint_img'] = ljoint_img
            out['joint_img'] = joint_img
            out['lmano_mesh_cam'] = lmesh_cam
            out['rmano_mesh_cam'] = rmesh_cam
            out['lmano_root_cam'] = lroot_cam
            out['rmano_root_cam'] = rroot_cam
            out['lmano_joint_cam'] = ljoint_cam
            out['rmano_joint_cam'] = rjoint_cam
            out['lmano_root_pose'] = lroot_pose
            out['rmano_root_pose'] = rroot_pose
            out['lmano_hand_pose'] = lhand_pose
            out['rmano_hand_pose'] = rhand_pose
            out['lmano_shape'] = lshape
            out['rmano_shape'] = rshape
            out['lmano_joint'] = ljoint_proj
            out['rmano_joint'] = rjoint_proj

            out['mano_pose'] = mano_pose
            out['mano_shape'] = mano_shape
            out['joint_cam'] = joint_cam
            out['mano_joint_cam'] = joint_cam
            out['joint_img'] = joint_img
            out['mano_joint_img'] = joint_img
            out['joint_proj'] = joint_proj
            out['rhand_feat'], out['lhand_feat'] = rhand_feat, lhand_feat

            if targets is not None and 'mano_joint_img' in targets:
                out['mano_joint_img'] = targets['mano_joint_img']
            if meta_info is not None and 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if targets is not None and 'mano_mesh_cam' in targets:
                out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
            if meta_info is not None and 'do_flip' in meta_info:
                out['do_flip'] = meta_info['do_flip']
            return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass

def get_rcvit_model(mode):
    # rcvit 384x384 -> 12 x 12 x 12 : feature dim -> 256: MACs 7.513G, Params 33.781M
    input_size = (1, 3, 256, 256)  # rcvit
    cfg.input_img_shape = (input_size[2], input_size[3])
    cfg.output_hm_shape = (input_size[2] // 32, input_size[3] // 32, input_size[3] // 32)
    cfg.hand_rcvit_type = 's'
    hand_backbone = RCViTBackboneNet(cfg.hand_rcvit_type)
    hand_position_net = PositionNet01(feat_dim=256, reduce_dim=128)
    hand_rotation_net = RotationNet01(in_chans=256, reduce_dim=128)

    if mode == 'train' and cfg.continue_train_kd == False and cfg.continue_train == False:
        hand_backbone.init_weights()
        hand_position_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)

    model = Model(hand_backbone, hand_position_net, hand_rotation_net)
    return model

def get_tinyvit_model(mode):
    # tinyvit 224x224 -> 7 x 7 x 7
    cfg.hand_tinyvit_type = '5m_ft'
    cfg.pretrained_path='./tiny_vit_5m_22kto1k_distill.pth'
    input_size = (1, 3, 224, 224)  # tinyvit_21m_ft
    cfg.input_img_shape = (input_size[2], input_size[3])
    cfg.output_hm_shape = (input_size[2] // 32, input_size[3] // 32, input_size[3] // 32)
    hand_backbone = TinyViTBackbone(cfg.hand_tinyvit_type, cfg.pretrained_path)
    hand_position_net = PositionNet01(feat_dim=320, reduce_dim=160)
    hand_rotation_net = RotationNet01(in_chans=320, reduce_dim=160)

    if mode == 'train' and cfg.continue_train_kd == False and cfg.continue_train == False:
        hand_backbone.init_weights()
        hand_position_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)

    model = Model(hand_backbone, hand_position_net, hand_rotation_net)
    return model

def get_resnet_model(mode):
    input_size = (1, 3, 256, 256)  # resnet50
    cfg.input_img_shape = (input_size[2], input_size[3])
    cfg.output_hm_shape = (input_size[2] // 32, input_size[3] // 32, input_size[3] // 32)
    hand_backbone = ResNetBackbone(cfg.hand_resnet_type)
    # resnet50 feature dim // 8 -> 256: MACs 7.652G, Params 52.870M
    hand_position_net = PositionNet()
    hand_rotation_net = RotationNet()

    if mode == 'train' and cfg.continue_train_kd == False and cfg.continue_train == False:
        hand_backbone.init_weights()
        hand_position_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)
    model = Model(hand_backbone, hand_position_net, hand_rotation_net)
    return model

def calculate_flops_and_params(model, input_size, modules=None):
    from thop import profile

    print("========================================================")

    if modules is None:
        modules = [
            ('hand_backbone', model.hand_backbone),
            ('hand_position_net', model.hand_position_net),
            ('hand_rotation_net', model.hand_rotation_net)
        ]

    input_img = torch.randn(input_size).cuda()
    flops_list = []
    params_list = []

    def profile_module(name, module, *inputs):
        flops, params = profile(module, inputs=inputs, verbose=False)

        flops_list.append(flops)
        params_list.append(params)
        print(f'{name.upper()}   MACs: {flops / 1e9} G')
        print(f'{name.upper()} Params: {params / 1e6} M')


    profile_module('hand_backbone', model.hand_backbone, input_img)
    feat = model.hand_backbone(input_img)


    profile_module('hand_position_net', model.hand_position_net, feat)
    rjoint_img, ljoint_img, rhand_feat, lhand_feat = model.hand_position_net(feat)

    profile_module('hand_rotation_net', model.hand_rotation_net, rhand_feat, lhand_feat, rjoint_img, ljoint_img)

    total_flops = sum(flops_list)
    total_params = sum(params_list)

    print(f'TOTAL   MACs: {total_flops / 1e9} G')
    print(f'TOTAL Params: {total_params / 1e6} M')

    return total_flops, total_params

def profile_model(model):

    torch.cuda.synchronize()


    batch_size = 1
    input_img = {'img': torch.randn(size=(batch_size, 3, 224, 224)).to('cuda')}
    model.cuda()
    model.eval()


    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_img)
            torch.cuda.synchronize()


    num_runs = 20
    num_iterations = 100
    latencies = []

    print("Testing performance...")
    for run in range(num_runs):
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(num_iterations):
                _ = model(input_img)

            torch.cuda.synchronize()
            end = time.perf_counter()


            run_latency = (end - start) / num_iterations
            latencies.append(run_latency)


            print(f"Run {run + 1}/{num_runs} completed")


    avg_latency = sum(latencies) / len(latencies)
    std_latency = np.std(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    avg_fps = 1 / avg_latency

    print("\nPerformance Report:")
    print(f"Average Latency: {avg_latency * 1000:.2f} ms")
    print(f"Latency Std Dev: {std_latency * 1000:.2f} ms")
    print(f"Min Latency: {min_latency * 1000:.2f} ms")
    print(f"Max Latency: {max_latency * 1000:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f} frames/second")


    print("\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

if __name__ == '__main__':
    from main.train import parse_args
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)


    input_size = (1, 3, 224, 224) # tinyvit_21m_ft
    cfg.input_img_shape = (input_size[2], input_size[3])
    cfg.output_hm_shape = (input_size[2] // 32, input_size[3] // 32, input_size[3] // 32)
    model = get_tinyvit_model(mode='eval')
    model.cuda()
    model.eval()
    calculate_flops_and_params(model, input_size)
    profile_model(model)



