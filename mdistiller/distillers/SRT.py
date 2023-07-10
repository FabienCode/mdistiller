import torch
import torch.nn as nn
import torch.nn.functional as F


from ._base import Distiller
from ..engine.kd_loss import KDQualityFocalLoss, kd_loss, dkd_loss
from ._common import ConvReg, ConvRegE, get_feat_shapes
# from mdistiller.models.transformer.model.decoder import Decoder
from mdistiller.engine.mdis_utils import CBAM
from torch.nn import Transformer
from mdistiller.engine.grad_cam_utils import GradCAM

class SRT(Distiller):
    """soft relaxation taylor approximation
    """

    def __init__(self, student, teacher, cfg):
        super(SRT, self).__init__(student, teacher)

        # vanilla kd setting
        self.temperature = cfg.KD.TEMPERATURE
        
        self.freeze(self.teacher)

        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        # add model

        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.warmup = cfg.DKD.WARMUP

        # self.cam = GradCAM(model=self.teacher, target_layers=self.teacher.layer3[-1], use_cuda=True)

        # self.hint_layer = 3
        # feat_s_shapes, feat_t_shapes = get_feat_shapes(
        #     self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        # )
        # self.conv_reg = ConvReg(
        #     feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        # )
        # self.channel_fusion = CBAM(512)
        # self.conv_reg = ConvRegE(512, 256)

        # self.cross_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8)
        # self.cross_module = nn.TransformerDecoder(self.cross_layer, num_layers=6)
        # self.cross_attention = nn.Transformer(d_model=64, nhead=4, num_encoder_layers=3, batch_first=True)


        # self.qkl_loss = KDQualityFocalLoss()

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        # teaacher have been freeze
        logits_teacher, feature_teacher = self.teacher(image)
        b, c, h, w = feature_teacher["feats"][-1].shape # 8 * 256 * 8 * 8

        # losses
        ce_loss_weight = 1
        loss_ce = ce_loss_weight * F.cross_entropy(logits_student, target)

        s_feat, t_feat = feature_student["feats"][-1], feature_teacher["feats"][-1]


        # weight = F.normalize(logits_teacher.pow(2).mean(0))
        weight = F.normalize(s_feat.reshape(b, c, -1).pow(2).mean(-1))
        # sorted_indices = torch.argsort(logits_teacher, dim=1)
        # top_length = int(logits_teacher.size(1) * 0.8)
        # top_indices = sorted_indices[:, :top_length]
        # mask = torch.ones_like(logits_teacher).scatter_(1, top_indices, 0).bool()
        sorted_indices = torch.argsort(weight, dim=1)
        top_length = int(weight.size(1) * 0.8)
        top_indices = sorted_indices[:, :top_length]
        mask = torch.ones_like(weight).scatter_(1, top_indices, 0).bool()
        other_mask = ~mask
        kd_loss_target = F.mse_loss(s_feat * mask.unsqueeze(-1).unsqueeze(-1), t_feat * mask.unsqueeze(-1).unsqueeze(-1))
        kd_loss_other = F.mse_loss(s_feat * other_mask.unsqueeze(-1).unsqueeze(-1), t_feat * other_mask.unsqueeze(-1).unsqueeze(-1))
        loss_kd = kd_loss_target + 8 * kd_loss_other

        # CrossKD
        # concat_feat = torch.concat((s_feat, t_feat), dim=1)
        # kd_feat = self.cross_attention(s_feat.reshape(b, c, -1), t_feat.reshape(b, c, -1)).reshape(b, c, h, w)
        # loss_kd = F.mse_loss(s_feat, kd_feat)
        # kd_feat = self.cross_attention(s_feat.reshape(b, c, -1), t_feat.reshape(b, c, -1)).reshape(b, c, h, w)

        # kd_feat = self.cross_attention(concat_feat.reshape(b, concat_feat.shape[1], -1), t_feat.reshape(b, c, -1)).reshape(b, c, h, w)
        # s_feat = self.channel_fusion(s_feat)
        # kd_feat = self.conv_reg(concat_feat)
        # kd_feat = self.custom_adaptive_pooling(kd_feat, 256)
        # kd_feat = self.cross_module(feature_student["feats"][-1].reshape(b, c, -1).permute(2,0,1), \
        #                             kd_feat.reshape(b, c, -1).permute(2,0,1)).permute(1,2,0).contiguous().reshape(b,c,h,w)
        # kd_feat = self.cross_module(s_feat.reshape(b, c, -1).permute(2,0,1), feature_teacher["feats"][-1]\
        #                             .reshape(b, c, -1).permute(2,0,1)).permute(1,2,0).contiguous().reshape(b,c,h,w)
        # kd_logits = self.teacher.fc(nn.AvgPool2d(h)(kd_feat).reshape(b, -1))

        # kd_loss_weight = 1
        # # min(kwargs["epoch"] / self.warmup, 1.0)
        # # loss_kd = kd_loss_weight * kd_loss(logits_student, kd_logits, self.kd_temperature) 
        # loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
        #     logits_student,
        #     logits_teacher,
        #     target,
        #     self.alpha,
        #     self.beta,
        #     self.temperature,
        #     mask
        # )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
    
    def align_scale(self, s_feat, t_feat): 
        N, C, H, W = s_feat.size()
        # normalize source feature
        s_feat = s_feat.permute(1, 0, 2, 3).reshape(C, -1)
        s_mean = s_feat.mean(dim=-1, keepdim=True)
        s_std = s_feat.std(dim=-1, keepdim=True)
        s_feat = (s_feat - s_mean) / (s_std + 1e-6)

        #
        t_feat = t_feat.permute(1, 0, 2, 3).reshape(C, -1)
        t_mean = t_feat.mean(dim=-1, keepdim=True)
        t_std = t_feat.std(dim=-1, keepdim=True)
        s_feat = s_feat * t_std + t_mean
        return s_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
    
    def custom_adaptive_pooling(self, input_tensor, output_channels):
        batch_size, input_channels, height, width = input_tensor.size()

        # 计算每个通道的最大池化输出
        pooled_tensor, _ = torch.max(input_tensor, dim=1, keepdim=True)

        # 扩展输出通道数
        pooled_tensor = pooled_tensor.expand(batch_size, output_channels, height, width)

        return pooled_tensor


# class CatReg(nn.Module):
#     def __init__(self, in_channel, out_channel, reduction_ratio=0.5, name=""):
#         super(CatReg, self).__init__()
#         self.name = name
#         self.reduction_ratio = reduction_ratio

#         hidden_num = in_channel

#         self.mlp_1_max = nn.Linear(hidden_num, int(hidden_num * reduction_ratio))
#         self.mlp_2_max = nn.Linear(int(hidden_num * reduction_ratio), hidden_num)
#         self.mlp_1_avg = nn.Linear(hidden_num, int(hidden_num * reduction_ratio))
#         self.mlp_2_avg = nn.Linear(int(hidden_num * reduction_ratio), hidden_num)
#         self.conv_layer = nn.Conv2d(2, 1, kernel_size=(7, 7), padding=(3, 3))
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
#         self.bn = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.conv_3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(out_channel)
        

#     def forward(self, inputs):
#         self.use_relu = True
#         batch_size, hidden_num = inputs.size(0), inputs.size(1)

#         maxpool_channel = torch.max(torch.max(inputs, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
#         avgpool_channel = torch.mean(torch.mean(inputs, dim=2, keepdim=True), dim=3, keepdim=True)

#         maxpool_channel = maxpool_channel.view(batch_size, -1)
#         avgpool_channel = avgpool_channel.view(batch_size, -1)

#         mlp_1_max = F.relu(self.mlp_1_max(maxpool_channel))
#         mlp_2_max = self.mlp_2_max(mlp_1_max).view(batch_size, 1, 1, hidden_num)

#         mlp_1_avg = F.relu(self.mlp_1_avg(avgpool_channel))
#         mlp_2_avg = self.mlp_2_avg(mlp_1_avg).view(batch_size, 1, 1, hidden_num)

#         channel_attention = torch.sigmoid(mlp_2_max + mlp_2_avg)
#         channel_refined_feature = inputs * channel_attention

#         maxpool_spatial = torch.max(inputs, dim=1, keepdim=True)[0]
#         avgpool_spatial = torch.mean(inputs, dim=1, keepdim=True)

#         max_avg_pool_spatial = torch.cat([maxpool_spatial, avgpool_spatial], dim=1)
#         spatial_attention = torch.sigmoid(self.conv_layer(max_avg_pool_spatial))

#         refined_feature = channel_refined_feature * spatial_attention

#         # conv reg
#         x = self.conv(refined_feature)
#         if self.use_relu:
#             x = self.relu(self.bn(x))
#         else:
#             x = self.bn(x)
#         x = self.conv_2(x)
#         if self.use_relu:
#             x = self.relu(self.bn2(x))
#         else:
#             x = self.bn2(x)
#         x = self.conv_3(x)
#         if self.use_relu:
#             return self.relu(self.bn3(x))
#         else:
#             return self.bn3(x)
#         # return refined_feature
