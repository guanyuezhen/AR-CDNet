import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_difference import TemporalFusion
from .encoder import FeatureFusionModule


class PredictionHead(nn.Module):
    def __init__(self, channel):
        super(PredictionHead, self).__init__()
        self.mask_generation = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1)
        )

    def forward(self, feature):
        mask = self.mask_generation(feature)

        return mask


class FeatureEnhancementUnit(nn.Module):
    def __init__(self, in_channel, channel):
        super(FeatureEnhancementUnit, self).__init__()
        self.feature_transition = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        hid_channel = max(8, channel // 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channel, hid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channel, in_channel),
            nn.Sigmoid()
        )
        self.feature_context = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask=None):
        x = self.feature_transition(x)
        if mask is not None:
            x = x * mask + x
        B, C, _, _ = x.size()
        vec_y = self.avg_pool(x).view(B, C)
        channel_att = self.channel_attention(vec_y).view(B, C, 1, 1)
        feu_out = self.feature_context(x * channel_att)

        return feu_out


class KnowledgeReviewModule(nn.Module):
    def __init__(self, in_channel, channel):
        super(KnowledgeReviewModule, self).__init__()
        #
        self.feu_1 = FeatureEnhancementUnit(in_channel, channel)
        self.feu_2 = FeatureEnhancementUnit(in_channel, channel)
        self.feu_3 = FeatureEnhancementUnit(in_channel, channel)
        #
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel + 1, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, fine_mask, coarse_mask):
        # without attention
        context_1 = self.feu_1(feature)
        # reverse attention
        reverse_att = 1 - torch.sigmoid(coarse_mask)
        context_2 = self.feu_2(feature, reverse_att)
        # uncertainty attention
        uncertainty_att = (1 - torch.sigmoid(coarse_mask)) * torch.sigmoid(fine_mask) + \
                          (1 - torch.sigmoid(fine_mask)) * torch.sigmoid(coarse_mask)
        context_3 = self.feu_3(feature, uncertainty_att)

        # import utils.torchutils as vis
        # vis.visulize_features(context_1)
        # vis.visulize_features(context_2)
        # vis.visulize_features(context_3)

        feature = context_1 + context_2 + context_3
        #
        krm_out = self.fusion_conv(torch.cat([feature, fine_mask], dim=1))

        # from PIL import Image
        # import numpy as np
        #
        # mask = torch.sigmoid(fine_mask)
        # mask = mask[0, 0].cpu().numpy() * 255
        # mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        # mask.save('/mnt/2800c818-54bc-4e2a-83d3-f418982b79e6/Change Detection/Methods_BCD/TDRNet/CDNet/maps/' + 'change_mask_p2.png')
        #
        # mask = torch.sigmoid(coarse_mask)
        # mask = mask[0, 0].cpu().numpy() * 255
        # mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        # mask.save('/mnt/2800c818-54bc-4e2a-83d3-f418982b79e6/Change Detection/Methods_BCD/TDRNet/CDNet/maps/' + 'change_mask_p3.png')
        #
        # mask = reverse_att
        # mask = mask[0, 0].cpu().numpy() * 255
        # mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        # mask.save('/mnt/2800c818-54bc-4e2a-83d3-f418982b79e6/Change Detection/Methods_BCD/TDRNet/CDNet/maps/' + 'reverse_att.png')
        #
        # mask = uncertainty_att
        # mask = mask[0, 0].cpu().numpy() * 255
        # mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        # mask.save('/mnt/2800c818-54bc-4e2a-83d3-f418982b79e6/Change Detection/Methods_BCD/TDRNet/CDNet/maps/' + 'uncertainty_att.png')

        return krm_out


class KnowledgeReviewDecoder(nn.Module):
    def __init__(self, channel):
        super(KnowledgeReviewDecoder, self).__init__()
        self.mask_generation_d2 = PredictionHead(channel)
        self.mask_generation_d3 = PredictionHead(channel)
        self.mask_generation_d4 = PredictionHead(channel)
        self.mask_generation_d5 = PredictionHead(channel)
        #
        self.krm_d3 = KnowledgeReviewModule(channel * 2, channel)
        self.krm_d4 = KnowledgeReviewModule(channel * 2, channel)
        self.krm_d5 = KnowledgeReviewModule(channel * 2, channel)
        #
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel * 5, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.mask_generation = PredictionHead(channel)

    def forward(self, d2, d3, d4, d5, b):
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        d4 = F.interpolate(d4, d2.size()[2:], mode='bilinear', align_corners=True)
        d5 = F.interpolate(d5, d2.size()[2:], mode='bilinear', align_corners=True)
        mask_d2 = self.mask_generation_d2(d2)
        mask_d3 = self.mask_generation_d3(d3)
        mask_d4 = self.mask_generation_d4(d4)
        mask_d5 = self.mask_generation_d5(d5)
        #
        d3 = self.krm_d3(torch.cat([d2, d3], dim=1), mask_d2, mask_d3)
        d4 = self.krm_d4(torch.cat([d2, d4], dim=1), mask_d2, mask_d4)
        d5 = self.krm_d5(torch.cat([d2, d5], dim=1), mask_d2, mask_d5)
        # d3 = self.krm_d3(torch.cat([d2, d3], dim=1), mask_d2.detach(), mask_d3.detach())
        # d4 = self.krm_d4(torch.cat([d2, d4], dim=1), mask_d2.detach(), mask_d4.detach())
        # d5 = self.krm_d5(torch.cat([d2, d5], dim=1), mask_d2.detach(), mask_d5.detach())
        #
        d_fusion = self.fusion_conv(torch.cat([d2, d3, d4, d5, b], dim=1))
        change_mask = self.mask_generation(d_fusion)

        # import utils.torchutils as vis
        # feature_vis = torch.cat([
        #     torch.mean(d2, dim=1, keepdim=True),
        #     torch.mean(d3, dim=1, keepdim=True),
        #     torch.mean(d4, dim=1, keepdim=True),
        #     torch.mean(d5, dim=1, keepdim=True),
        #     torch.mean(b, dim=1, keepdim=True)
        # ], dim=1)
        # vis.visulize_features(feature_vis)

        return change_mask, mask_d2, mask_d3, mask_d4, mask_d5


class Boundary_Decoder(nn.Module):
    def __init__(self, channel):
        super(Boundary_Decoder, self).__init__()
        self.feature_texture = nn.Sequential(
            nn.Conv2d(3, channel // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.td = TemporalFusion(channel)
        self.fam = FeatureFusionModule(channel)
        self.boundary_generation = PredictionHead(channel)

    def forward(self, t1, t2, d5):
        t1_texture = self.feature_texture(t1)
        t2_texture = self.feature_texture(t2)

        boundary = self.td(t1_texture, t2_texture)

        boundary = self.fam(d5, boundary)

        # import utils.torchutils as vis
        # vis.visulize_features(t1_texture)
        # vis.visulize_features(t2_texture)
        # vis.visulize_features(boundary)

        boundary_mask = self.boundary_generation(boundary)

        return boundary_mask, boundary

