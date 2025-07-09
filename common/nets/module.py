import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from common.nets.layer import make_conv_layers, DyT, make_deconv_layers, make_linear_layers
from common.utils.human_models import mano
from common.utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from main.config import cfg
from common.nets.crosstransformer import CrossTransformer
# from common.nets.crosstransformer import Block
from einops import rearrange
from timm.models.vision_transformer import Block
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeftRightStream(nn.Module):
    def __init__(self, in_channels):
        super(LeftRightStream, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, x):
        gap = torch.mean(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([gap, gmp], dim=1)
        y = self.conv(y)
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        y = avg_out + max_out
        return y

class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)

    def forward(self, x, y_f):
        y = torch.cat([x, y_f], dim=1)
        y = self.conv1(y)
        y = self.conv2(y)
        return y

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out

class FuseAttention(nn.Module):
    def __init__(self, in_channels):
        super(FuseAttention, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(in_channels)
        self.pixel_attention = PixelAttention(in_channels)
        self.cross_attention = CrossAttention(in_channels)

    def forward(self, x):
        y_s = self.spatial_attention(x)
        y_c = self.channel_attention(x)
        # fs = torch.sigmoid(y_s) * x
        # fc = torch.sigmoid(y_c) * x
        fs = torch.sigmoid(y_s * x)
        fc = torch.sigmoid(y_c * x)
        y_f = fs + fc
        # y_f = fc
        y_p = self.pixel_attention(x, y_f)
        y_ca = self.cross_attention(y_p)
        return y_ca

class AdaptiveFeatureEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveFeatureEnhancement, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inter_feat, hand_feat):
        fused_feat = torch.cat([inter_feat, hand_feat], dim=1)
        fused_feat = self.conv(fused_feat)
        return fused_feat

class PEModule(nn.Module):
    def __init__(self, in_channels):
        super(PEModule, self).__init__()
        self.left_stream = LeftRightStream(in_channels)
        self.right_stream = LeftRightStream(in_channels)
        self.fuse_attention = FuseAttention(in_channels)
        self.adaptive_enhancement_left = AdaptiveFeatureEnhancement(in_channels)
        self.adaptive_enhancement_right = AdaptiveFeatureEnhancement(in_channels)

    def forward(self, x):
        left_feat = self.left_stream(x)
        right_feat = self.right_stream(x)

        mid_feat_left = self.fuse_attention(left_feat)
        mid_feat_right = self.fuse_attention(right_feat)
        # left_feat, right_feat = mid_feat_left, mid_feat_right
        #
        enhanced_left = self.adaptive_enhancement_left(mid_feat_left, left_feat)
        enhanced_right = self.adaptive_enhancement_right(mid_feat_right, right_feat)

        # return left_feat, right_feat
        return enhanced_left, enhanced_right, left_feat, right_feat

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.patience_counter = 0

    def check_early_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False  # No early stop
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True  # Early stop
            else:
                return False  # No early stop


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, hidden_size, out_channel, padding, dilation):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,  # 220
                      out_channels=hidden_size,  # 128*2=256
                      kernel_size=3,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      bias=False,
                      groups=hidden_size),  # 128*2=256 groups=hidden_size
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=out_channel,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False,
                      groups=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class FuseFormer(nn.Module):
    def __init__(self, in_chans=512, num_token=12, depth=4, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(FuseFormer, self).__init__()
        self.FC = nn.Linear(in_chans * 2, in_chans)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + (2 * num_token * num_token), in_chans))
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_chans))
        self.SA_T = nn.ModuleList([
            Block(in_chans, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            # Block_new(in_chans, num_heads, mlp_ratio, qkv_bias=False)
            for i in range(depth)])
        self.FC2 = nn.Linear(in_chans, in_chans)
        # Decoder
        self.CA_T = CrossTransformer(in_chans1=in_chans, in_chans2=in_chans, num_token=num_token)
        self.FC3 = nn.Linear(in_chans, in_chans)

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        feat1 = rearrange(feat1, 'B C H W -> B (H W) C')
        feat2 = rearrange(feat2, 'B C H W -> B (H W) C')

        # joint Token
        token_j = self.FC(torch.cat((feat1, feat2), dim=-1))

        # similar token
        token_s = torch.cat((feat1, feat2), dim=1) + self.pos_embed[:, 1:]
        cls_token = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        token_s = torch.cat((cls_token, token_s), dim=1)
        for blk in self.SA_T:
            token_s = blk(token_s)
        token_s = self.FC2(token_s)

        output = self.CA_T(token_j, token_s)
        # output, attn = self.CA_T(token_j, token_s)
        output = self.FC3(output)
        output = rearrange(output, 'B (H W) C -> B C H W', H=H, W=W)
        # return output, attn
        return output
        # return output, token_s, token_j


class EABlock(nn.Module):
    def __init__(self, feat_dim=2048, hidden_size=512, num_token=12):
        super(EABlock, self).__init__()
        self.PEModule = PEModule(feat_dim)

    def forward(self, hand_feat):
        rhand_feat, lhand_feat, left_feat, right_feat= self.PEModule(hand_feat)

        return rhand_feat, lhand_feat, left_feat, right_feat


class PositionNet(nn.Module):
    def __init__(self, feat_dim=2048, reduce_dim=512):
        super(PositionNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.EABlock = EABlock(feat_dim=feat_dim, hidden_size=reduce_dim, num_token=cfg.output_hm_shape[2])
        self.conv_r2 = make_conv_layers([feat_dim, self.joint_num * cfg.output_hm_shape[2]], kernel=1, stride=1,
                                        padding=0, bnrelu_final=False)
        self.conv_l2 = make_conv_layers([feat_dim, self.joint_num * cfg.output_hm_shape[2]], kernel=1, stride=1,
                                        padding=0, bnrelu_final=False)

    def forward(self, hand_feat):
        rhand_feat, lhand_feat, left_feat, right_feat = self.EABlock(hand_feat)
        rhand_hm = self.conv_r2(rhand_feat)
        rhand_hm = rhand_hm.view(-1, self.joint_num, cfg.output_hm_shape[2], cfg.output_hm_shape[0],
                                 cfg.output_hm_shape[1])
        rhand_coord = soft_argmax_3d(rhand_hm)

        lhand_hm = self.conv_l2(lhand_feat)
        lhand_hm = lhand_hm.view(-1, self.joint_num, cfg.output_hm_shape[2], cfg.output_hm_shape[0],
                                 cfg.output_hm_shape[1])
        lhand_coord = soft_argmax_3d(lhand_hm)

        return rhand_coord, lhand_coord, rhand_feat, lhand_feat
        # return rhand_coord, lhand_coord, rhand_feat, lhand_feat, left_feat, right_feat


class Transformer(nn.Module):
    def __init__(self, in_chans=512, joint_num=21, depth=4, num_heads=8, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, joint_num, in_chans))
        self.blocks = nn.ModuleList([
            Block(in_chans, num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x


class RotationNet(nn.Module):
    def __init__(self, in_chans=1024, reduce_dim=512):
        super(RotationNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.rconv = make_conv_layers([in_chans, reduce_dim], kernel=1, stride=1, padding=0)
        self.lconv = make_conv_layers([in_chans, reduce_dim], kernel=1, stride=1, padding=0)
        self.rshape_out = make_linear_layers([in_chans, mano.shape_param_dim], relu_final=False)
        self.rcam_out = make_linear_layers([in_chans, 3], relu_final=False)
        self.lshape_out = make_linear_layers([in_chans, mano.shape_param_dim], relu_final=False)
        self.lcam_out = make_linear_layers([in_chans, 3], relu_final=False)


        # relative translation
        self.root_relative = make_linear_layers([2 * in_chans, reduce_dim, 3], relu_final=False)


        self.rroot_pose_out = make_linear_layers([self.joint_num * (reduce_dim + 3), 6], relu_final=False)
        self.rpose_out = make_linear_layers([self.joint_num * (reduce_dim + 3), (mano.orig_joint_num - 1) * 6],
                                            relu_final=False)  # without root joint
        self.lroot_pose_out = make_linear_layers([self.joint_num * (reduce_dim + 3), 6], relu_final=False)
        self.lpose_out = make_linear_layers([self.joint_num * (reduce_dim + 3), (mano.orig_joint_num - 1) * 6],
                                            relu_final=False)  # without root joint

    def forward(self, rhand_feat, lhand_feat, rjoint_img, ljoint_img):
        batch_size = rhand_feat.shape[0]
        # shape and camera parameters
        rshape_param = self.rshape_out(rhand_feat.mean((2, 3)))
        rcam_param = self.rcam_out(rhand_feat.mean((2, 3)))
        lshape_param = self.lshape_out(lhand_feat.mean((2, 3)))
        lcam_param = self.lcam_out(lhand_feat.mean((2, 3)))
        rel_trans = self.root_relative(torch.cat((rhand_feat, lhand_feat), dim=1).mean((2, 3)))

        # xyz corrdinate feature
        rhand_feat = self.rconv(rhand_feat)
        lhand_feat = self.lconv(lhand_feat)
        rhand_feat = sample_joint_features(rhand_feat, rjoint_img[:, :, :2])  # batch_size, joint_num, feat_dim
        lhand_feat = sample_joint_features(lhand_feat,
                                           ljoint_img[:, :, :2])  # batch_size, joint_num, feat_dim JointFeatureSampler

        # Relative Translation
        rhand_feat = torch.cat((rhand_feat, rjoint_img), 2).view(batch_size, -1)
        lhand_feat = torch.cat((lhand_feat, ljoint_img), 2).view(batch_size, -1)
        rhand_feat = rhand_feat.view(batch_size, -1)
        lhand_feat = lhand_feat.view(batch_size, -1)

        rroot_pose = self.rroot_pose_out(rhand_feat)
        rpose_param = self.rpose_out(rhand_feat)
        lroot_pose = self.lroot_pose_out(lhand_feat)
        lpose_param = self.lpose_out(lhand_feat)

        return rroot_pose, rpose_param, rshape_param, rcam_param, lroot_pose, lpose_param, lshape_param, lcam_param, rel_trans

