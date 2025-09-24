import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer
import math


def make_grid(grid_shape):
    
    #grid: (y, x, z)
    grid_1ds = [torch.arange(-1, 1, 2.0/shape) for shape in grid_shape]
    grids = torch.meshgrid(grid_1ds)
    return grids

class CostVolume(nn.Module):
    """
        While PSV module define depth dimension similar to the depth in real world

        Cost Volume implementation in PSM network and its prior networks define this directly as disparity
    """
    def __init__(self, max_disp=192, downsample_scale=4, input_features=1024, PSM_features=64):
        super(CostVolume, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)
        self.down_sample = nn.Sequential(
            nn.Conv2d(input_features, PSM_features, 1),
            nn.BatchNorm2d(PSM_features),
            nn.ReLU(),
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(2 * PSM_features, PSM_features, 3, padding=1),
            nn.BatchNorm3d(PSM_features),
            nn.ReLU(),
            nn.Conv3d(PSM_features, PSM_features, 3, padding=1),
            nn.BatchNorm3d(PSM_features),
            nn.ReLU(),
        )
        self.output_channel = PSM_features * self.depth_channel

    def forward(self, left_features, right_features):
        batch_size, _, w, h = left_features.shape
        left_features = self.down_sample(left_features)
        right_features = self.down_sample(right_features)
        
        if not self.training:
            with torch.no_grad():
                cost = torch.FloatTensor(left_features.size()[0],
                                left_features.size()[1]*2,
                                self.depth_channel,
                                left_features.size()[2],  
                                left_features.size()[3]).zero_().cuda()
        else:
            cost = torch.FloatTensor(left_features.size()[0],
                            left_features.size()[1]*2,
                            self.depth_channel,
                            left_features.size()[2],  
                            left_features.size()[3]).zero_().cuda()

        for i in range(self.depth_channel):
            if i > 0 :
                 cost[:, :left_features.size()[1], i, :,i:]  = left_features[:,:,:,i:]
                 cost[:, left_features.size()[1]:, i, :,i:]  = right_features[:,:,:,:-i]
            else:
                 cost[:, :left_features.size()[1], i, :,:]   = left_features
                 cost[:, left_features.size()[1]:, i, :,:]   = right_features
        cost = cost.contiguous()
        cost = self.conv3d(cost) # .squeeze(1)
        cost = cost.reshape(batch_size, -1, w, h).contiguous()
        return cost


class PSMCosineModule(nn.Module):
    """Some Information about PSMCosineModule"""
    def __init__(self, max_disp=192, downsample_scale=4, input_features=512):
        super(PSMCosineModule, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)

    def forward(self, left_features, right_features):
       
        if not self.training:
            with torch.no_grad():
                cost = torch.FloatTensor(left_features.size()[0],
                              self.depth_channel,
                              left_features.size()[2],  
                              left_features.size()[3]).zero_().cuda()
        else:
             cost = torch.FloatTensor(left_features.size()[0],
                              self.depth_channel,
                              left_features.size()[2],  
                              left_features.size()[3]).zero_().cuda()

        for i in range(self.depth_channel):
            if i > 0 :
                 cost[:, i, :,i:]  = (left_features[:,:,:,i:] * right_features[:,:,:,:-i]).mean(dim=1)
            else:
                 cost[:, i, :, :]  = (left_features * right_features).mean(dim=1)
        cost = cost.contiguous()
        return cost

class DoublePSMCosineModule(PSMCosineModule):
    """Some Information about DoublePSMCosineModule"""
    def __init__(self, max_disp=192, downsample_scale=4):
        super(DoublePSMCosineModule, self).__init__(max_disp=max_disp, downsample_scale=downsample_scale)
        self.depth_channel = self.depth_channel

    def forward(self, left_features, right_features):
        b, c, h, w = left_features.shape
        base_grid_y, base_grid_x = make_grid(right_features.shape[2:]) #[h, w]
        base_grid_x = base_grid_x - 1.0 / right_features.shape[1] 
        shifted_grid = torch.stack([base_grid_y, base_grid_x], dim=-1).cuda().unsqueeze(0).repeat(b, 1, 1, 1)
        right_features_shifted = F.grid_sample(right_features, shifted_grid)
        cost_1 = super(DoublePSMCosineModule, self)(left_features, right_features)
        cost_2 = super(DoublePSMCosineModule, self)(left_features, right_features_shifted)
        return torch.cat([cost_1, cost_2], dim=1)


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
    
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CostVolumePyramid(nn.Module):
    """Some Information about CostVolumePyramid"""
    def __init__(self, depth_channel_8, depth_channel_16):
        super(CostVolumePyramid, self).__init__()
        self.depth_channel_8  = depth_channel_8 # 24
        self.depth_channel_16 = depth_channel_16 # 96

        input_features = depth_channel_8 # 3 * 24 + 24 = 96
        self.eight_to_sixteen = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features)
        )
        input_features = 3 * input_features + depth_channel_16 # 3 * 96 + 96 = 384
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            BasicBlock(3 * input_features, 3 * input_features),
        )
        self.output_channel_num = 3 * input_features #1152
        self.depth_output = nn.Sequential(
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

    def forward(self, psv_volume_8, psv_volume_16):
        # psv_4_8 = self.four_to_eight(psv_volume_4)
        # psv_volume_8 = torch.cat([psv_4_8, psv_volume_8], dim=1)
        psv_8_16 = self.eight_to_sixteen(psv_volume_8)
        psv_volume_16 = torch.cat([psv_8_16, psv_volume_16], dim=1)
        psv_16 = self.depth_reason(psv_volume_16)
        if self.training:
            return psv_16, self.depth_output(psv_16)
        return psv_16, torch.zeros([psv_volume_8.shape[0], 1, psv_volume_8.shape[2], psv_volume_8.shape[3]])

class StereoDepthPredictor(nn.Module):

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
        self.upsample = nn.Sequential(
            nn.Conv2d(504, 256, kernel_size=(1, 1)),
            nn.GroupNorm(32, 256))

        self.depth_head = nn.Sequential(
            nn.Conv2d(256, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1)

        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)
        self.cost_volume_1 = PSMCosineModule(downsample_scale=8, max_disp=192, input_features=256)
        PSV_depth_1 = self.cost_volume_1.depth_channel
        self.cost_volume_2 = CostVolume(downsample_scale=16, max_disp=192, input_features=256, PSM_features=8)
        PSV_depth_2 = self.cost_volume_2.output_channel
        self.depth_reasoning = CostVolumePyramid(PSV_depth_1, PSV_depth_2)
    def forward(self, feature_stereo, mask, pos):
       
        assert len(feature_stereo) == 4
        batch_size_half = feature_stereo[0].shape[0] //2
        # foreground depth map
        #ipdb.set_trace()
        src_16 = self.proj(feature_stereo[1])
   
        PSVolume_1 = self.cost_volume_1(feature_stereo[0][:batch_size_half], feature_stereo[0][batch_size_half:])
        PSVolume_2 = self.cost_volume_2(feature_stereo[1][:batch_size_half], feature_stereo[1][batch_size_half:])
        PSV_features, pred_disp = self.depth_reasoning(PSVolume_1, PSVolume_2) # c = 1152
        PSV_features_inter = self.upsample(F.interpolate(PSV_features, size=src_16.shape[-2:], mode='bilinear'))
 
        features_for_depth = PSV_features_inter
        src = self.depth_head(features_for_depth)
        #ipdb.set_trace()
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)
        #ipdb.set_trace()
        # depth embeddings with depth positional encodings
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)
        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
        #ipdb.set_trace()
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip

        return depth_logits, depth_embed, weighted_depth, depth_pos_embed_ip, PSV_features_inter, pred_disp

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
