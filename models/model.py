import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import KnowledgeReviewDecoder
from .decoder import Boundary_Decoder
from .temporal_difference import TemporalFusion


class BaseNet(nn.Module):
    def __init__(self, input=3, output=1):
        super(BaseNet, self).__init__()
        channel = 64
        self.encoder = Encoder(channel)
        self.td_d2 = TemporalFusion(channel)
        self.td_d3 = TemporalFusion(channel)
        self.td_d4 = TemporalFusion(channel)
        self.td_d5 = TemporalFusion(channel)
        self.kr_decoder = KnowledgeReviewDecoder(channel)
        self.b_decoder = Boundary_Decoder(channel)

    def forward(self, x):
        t1 = x[:, 0:3, :, :]
        t2 = x[:, 3:6, :, :]
        #
        t1_p2, t1_p3, t1_p4, t1_p5 = self.encoder(t1)
        t2_p2, t2_p3, t2_p4, t2_p5 = self.encoder(t2)
        #
        d2 = self.td_d2(t1_p2, t2_p2)
        d3 = self.td_d3(t1_p3, t2_p3)
        d4 = self.td_d4(t1_p4, t2_p4)
        d5 = self.td_d5(t1_p5, t2_p5)

        # import tools.torchutils as vis
        # feature_vis = torch.cat([
        #     F.interpolate(torch.mean(d5, dim=1, keepdim=True), d2.size()[2:], mode='bilinear', align_corners=True),
        #     F.interpolate(torch.mean(d4, dim=1, keepdim=True), d2.size()[2:], mode='bilinear', align_corners=True),
        #     F.interpolate(torch.mean(d3, dim=1, keepdim=True), d2.size()[2:], mode='bilinear', align_corners=True),
        #     torch.mean(d2, dim=1, keepdim=True)
        # ], dim=1)
        # vis.visulize_features(feature_vis)

        # boundary learning
        boundary_mask, boundary_feature = self.b_decoder(t1, t2, d2)
        boundary_mask = F.interpolate(boundary_mask, x.size()[2:], mode='bilinear', align_corners=True)
        boundary_mask = torch.sigmoid(boundary_mask)

        # knowledge review
        change_mask, mask_d2, mask_d3, mask_d4, mask_d5 = self.kr_decoder(d2, d3, d4, d5, boundary_feature)
        change_mask = F.interpolate(change_mask, x.size()[2:], mode='bilinear', align_corners=True)
        mask_d2 = F.interpolate(mask_d2, x.size()[2:], mode='bilinear', align_corners=True)
        mask_d3 = F.interpolate(mask_d3, x.size()[2:], mode='bilinear', align_corners=True)
        mask_d4 = F.interpolate(mask_d4, x.size()[2:], mode='bilinear', align_corners=True)
        mask_d5 = F.interpolate(mask_d5, x.size()[2:], mode='bilinear', align_corners=True)
        change_mask = torch.sigmoid(change_mask)
        mask_d2 = torch.sigmoid(mask_d2)
        mask_d3 = torch.sigmoid(mask_d3)
        mask_d4 = torch.sigmoid(mask_d4)
        mask_d5 = torch.sigmoid(mask_d5)
        #

        # import utils.torchutils as vis
        # vis.visulize_features(torch.cat([
        #     change_mask, mask_d2, mask_d3, mask_d4, mask_d5, boundary_mask
        # ], dim=1))

        return change_mask, mask_d2, mask_d3, mask_d4, mask_d5, boundary_mask


def get_model():
    model = BaseNet(3, 1)

    return model

