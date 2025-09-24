import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer
import math


class MobileV2Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        # v2
        self.pwconv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        # v2
        feat = self.pwconv(x)
        feat = self.dwconv(feat)
        feat = self.pwliner(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat


class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost

class GhostModule(nn.Module):
    """
        Ghost Module from https://github.com/iamhankai/ghostnet.pytorch.

    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Sequential(),
            nn.Conv2d(inp, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class ResGhostModule(GhostModule):
    """Some Information about ResGhostModule"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, relu=True, stride=1):
        assert(ratio > 2)
        super(ResGhostModule, self).__init__(inp, oup-inp, kernel_size, ratio-1, dw_size, relu=relu, stride=stride)
        self.oup = oup
        if stride > 1:
            self.downsampling = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsampling = None

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        if not self.downsampling is None:
            x = self.downsampling(x)
        out = torch.cat([x, x1, x2], dim=1)
        return out[:,:self.oup,:,:]


class Aggregation(nn.Module):
    def __init__(self, in_channels, in_channels_s8, in_channels_s16, left_att, blocks, expanse_ratio, backbone_channels):
        super(Aggregation, self).__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio[0]
        self.expanse_ratio_8 = expanse_ratio[1]
        self.expanse_ratio_16 = expanse_ratio[2]

        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        conv0_8 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio_8)
                 for i in range(blocks[0])]
        self.conv0_8 = nn.Sequential(*conv0_8)

        conv0_16 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio_16)
                 for i in range(blocks[0])]
        self.conv0_16 = nn.Sequential(*conv0_16)

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_add = [MobileV2Residual(in_channels * 2 + in_channels_s8, 
                                      in_channels * 2 + in_channels_s8, 
                                      stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[1] - 1)]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = MobileV2Residual(in_channels * 2 + in_channels_s8, 
                                      in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_add = [MobileV2Residual(in_channels * 4 + in_channels_s16, 
                                      in_channels * 4 + in_channels_s16, 
                                      stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[2] - 1)]
        self.conv4 = nn.Sequential(*conv4_add)

        if self.left_att:
            self.att0 = AttentionModule(in_channels, backbone_channels[0])
            self.att2 = AttentionModule(in_channels * 2 + in_channels_s8, backbone_channels[1])
            self.att4 = AttentionModule(in_channels * 4 + in_channels_s16, backbone_channels[2])
        input_features = in_channels * 4 + in_channels_s16    
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            MobileV2Residual(3 * input_features, 
                             512, stride=1, expanse_ratio=self.expanse_ratio)
        )
    def forward(self, x, x_8, x_16, features_left):

        x = self.conv0(x)
        if self.left_att:
            x = self.att0(x, features_left[0])
        conv1 = self.conv1(x) # b, 48, 36, 160
       
        conv1 = torch.concat([x_8, conv1], dim=1) # b, 48+24, 36, 160
        conv2 = self.conv2(conv1)
    
        if self.left_att:
            conv2 = self.att2(conv2, features_left[1])
        conv3 = self.conv3(conv2)
        
        conv3 = torch.concat([x_16, conv3], dim=1) # b, 48+24+12, 36, 160
      
        conv4 = self.conv4(conv3)
        
        if self.left_att:
            conv4 = self.att4(conv4, features_left[2])
        conv5 = self.depth_reason(conv4)
       
        return conv5


def correlation_volume(left_feature, right_feature, max_disp):
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume

def correlation_volume_flip(left_feature, right_feature, max_disp):
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, :-i] * right_feature[:, :, :, i:]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume

class LightStereoDepthPredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"])
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"])
        self.depth_max = depth_max
        self.decoder_type = model_cfg['decoder_type']

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())
        # self.dims_head = nn.Sequential(
        #     nn.Conv2d(512, d_model, kernel_size=(3, 3), padding=1),
        #     nn.GroupNorm(32, num_channels=d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
        #     nn.GroupNorm(32, num_channels=d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(d_model, 1, kernel_size=(1, 1)))
        
        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))
        
        self.output_channel_num = 256
        self.disp_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 96, 1),
        )
        if self.decoder_type == "depthaware":
            depth_encoder_layer = TransformerEncoderLayer(
                d_model, nhead=8, dim_feedforward=256, dropout=0.1)

            self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

            self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)
        

        # aggregation
        self.cost_agg = Aggregation(in_channels=24,
                                    in_channels_s8=24, #24
                                    in_channels_s16=12,
                                    left_att=True,
                                    blocks=[ 1, 2, 4 ],
                                    expanse_ratio=[4, 4, 4],
                                    backbone_channels=[ 256, 256, 256 ])
        
    def forward(self, feature_stereo, mask, pos, targets=None):
       
        assert len(feature_stereo) == 4
        batch_size_half = feature_stereo[0].shape[0] //2
        # foreground depth map
        #ipdb.set_trace()
        # src_16 = self.proj(feature_stereo[2])
        # src_16_left = src_16[:batch_size_half]
        # src_32 = self.upsample(F.interpolate(feature_stereo[2][:batch_size_half], size=src_16.shape[-2:], mode='bilinear'))
        # src_8 = self.downsample(feature_stereo[1][:batch_size_half])
        if self.training:
            gwc_volume_list4 = []
            gwc_volume_list8 = []
            gwc_volume_list16 = []
            for  batch_id in range(batch_size_half):
                flip_flag = targets[batch_id]["random_flip_flag"]
                switch_flag = targets[batch_id]["random_switch_flag"]
                f_left_i_s4 = feature_stereo[0][batch_id].unsqueeze(0)
                f_right_i_s4 = feature_stereo[0][batch_id+batch_size_half].unsqueeze(0)
                f_left_i_s8 = feature_stereo[1][batch_id].unsqueeze(0)
                f_right_i_s8 = feature_stereo[1][batch_id+batch_size_half].unsqueeze(0)
                f_left_i_s16 = feature_stereo[2][batch_id].unsqueeze(0)
                f_right_i_s16 = feature_stereo[2][batch_id+batch_size_half].unsqueeze(0)
                if flip_flag or switch_flag:
                    gwc_volume_list4.append(correlation_volume_flip(f_left_i_s4, f_right_i_s4, 96 // 4)) 
                    gwc_volume_list8.append(correlation_volume_flip(f_left_i_s8, f_right_i_s8, 192 // 8))
                    gwc_volume_list16.append(correlation_volume_flip(f_left_i_s16, f_right_i_s16, 192 // 16))
                else:
                    gwc_volume_list4.append(correlation_volume(f_left_i_s4, f_right_i_s4, 96 // 4))
                    gwc_volume_list8.append(correlation_volume(f_left_i_s8, f_right_i_s8, 192 // 8)) # 192/8
                    gwc_volume_list16.append(correlation_volume(f_left_i_s16, f_right_i_s16, 192 // 16))
            gwc_volume_s4 = torch.cat(gwc_volume_list4, 0)
            gwc_volume_s8 = torch.cat(gwc_volume_list8, 0)
            gwc_volume_s16 = torch.cat(gwc_volume_list16, 0)
        else:      
            gwc_volume_s4 = correlation_volume(feature_stereo[0][:batch_size_half], 
                                               feature_stereo[0][batch_size_half:], 
                                               96 // 4)
            gwc_volume_s8 = correlation_volume(feature_stereo[1][:batch_size_half], 
                                               feature_stereo[1][batch_size_half:], 
                                               192 // 8)  # 192/8
            gwc_volume_s16 = correlation_volume(feature_stereo[2][:batch_size_half], 
                                               feature_stereo[2][batch_size_half:], 
                                               192 // 16)
        features_left = []
        for i in range(len(feature_stereo)):
            # features_left.append(feature_stereo[i][batch_size_half:])
            features_left.append(feature_stereo[i][:batch_size_half])
        PSV_features = self.cost_agg(gwc_volume_s4, gwc_volume_s8, gwc_volume_s16, features_left)
        features_for_depth = PSV_features
        src = self.depth_head(features_for_depth)
        
        #ipdb.set_trace()
        if self.training:
            pred_disp = self.disp_output(src)
            # dims_pre = self.dims_head(features_for_depth)
        else:
            pred_disp = None
            # dims_pre = None
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)
        #ipdb.set_trace()
        # depth embeddings with depth positional encodings
        if self.decoder_type == "depthaware":
            B, C, H, W = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
            pos = pos.flatten(2).permute(2, 0, 1)
            depth_embed = self.depth_encoder(src, mask, pos)
            depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
            #ipdb.set_trace()
            depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
            depth_embed = depth_embed + depth_pos_embed_ip
        else:
            depth_embed = None
            depth_pos_embed_ip = None

        return depth_logits, depth_embed, weighted_depth, depth_pos_embed_ip, PSV_features, pred_disp

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
