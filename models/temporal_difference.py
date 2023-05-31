import torch
import torch.nn as nn


class TemporalFusion(nn.Module):
    def __init__(self, channel):
        super(TemporalFusion, self).__init__()
        self.conv_context_t1_to_t2 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=[2, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1]),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True)
        )
        self.conv_context_t2_to_t1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=[2, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1]),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1, t2):
        B, C, H, W = t1.size()
        t1 = t1.view(B, C, 1, H, W)
        t2 = t2.view(B, C, 1, H, W)
        t1_to_t2 = torch.cat([t1, t2], dim=2)
        t2_to_t1 = torch.cat([t1, t2], dim=2)
        diff_1 = self.conv_context_t1_to_t2(t1_to_t2).view(B, C, H, W)
        diff_2 = self.conv_context_t2_to_t1(t2_to_t1).view(B, C, H, W)
        diff = diff_1 + diff_2

        # import utils.torchutils as vis
        # feature_vis = torch.cat([
        #     torch.mean(diff_1, dim=1, keepdim=True),
        #     torch.mean(diff_2, dim=1, keepdim=True)
        # ], dim=1)
        # vis.visulize_features(feature_vis)

        return diff
