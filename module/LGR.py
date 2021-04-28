import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from module.GCN import GCN
import math
import torch.nn.functional as F

device_id = 0

def L_Matrix(adj_npy, adj_size):

    D =np.zeros((adj_size, adj_size))
    for i in range(adj_size):
        tmp = adj_npy[i,:]
        count = np.sum(tmp==1)
        if count>0:
            number = count ** (-1/2)
            D[i,i] = number

    x = np.matmul(D,adj_npy)
    L = np.matmul(x,D)
    return L


class SEQuart(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, dim_d):
        super(SEQuart, self).__init__()
        dim = dim_a + dim_b + dim_c + dim_d

        self.gcn = GCN(4, 64, 64)

        self.fc_one = nn.Sequential(
            nn.Linear(dim, dim_a),
            nn.Sigmoid()
        )
        self.fc_two = nn.Sequential(
            nn.Linear(dim, dim_b),
            nn.Sigmoid()
        )
        self.fc_three = nn.Sequential(
            nn.Linear(dim, dim_c),
            nn.Sigmoid()
        )
        self.fc_four = nn.Sequential(
            nn.Linear(dim, dim_c),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        # self.gate_a = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_b = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_c = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_d = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(dim, 64, kernel_size=1, bias=True)

    def forward(self, low, high, flow, feedback):
        batch, channel, _, _ = low.size()
        combined = torch.cat([low, high, flow, feedback], dim=1)
        combined_fc = self.avg_pool(combined).view(batch, 4, channel)

        combined_fc_norm = torch.norm(combined_fc, dim=2, keepdim=True)
        combined_fc_norm_t = combined_fc_norm.permute(0, 2, 1)
        combined_fc_t = combined_fc.permute(0, 2, 1)
        mul = torch.bmm(combined_fc, combined_fc_t)
        batch_adj = mul / (combined_fc_norm * combined_fc_norm_t)


        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)
        excitation1 = self.fc_one(feat_cat).view(batch, channel, 1, 1)
        excitation2 = self.fc_two(feat_cat).view(batch, channel, 1, 1)
        excitation3 = self.fc_three(feat_cat).view(batch, channel, 1, 1)
        excitation4 = self.fc_four(feat_cat).view(batch, channel, 1, 1)
        
        feat_cat = torch.cat([excitation1 * low, excitation2 * high, excitation3 * flow, excitation4 * feedback], dim=1)

        attention_vector = self.gate(feat_cat)

        return attention_vector

class SEMany2Many3(nn.Module):
    def __init__(self, many, many2, dim_one):
        super(SEMany2Many3, self).__init__()

        self.gcn = GCN(many, dim_one, dim_one)

        self.fc_one = nn.Sequential(
            nn.Linear(many * dim_one, 2 * dim_one),
            nn.Sigmoid()
        )
        self.fc_two = nn.Sequential(
            nn.Linear(many * dim_one, 2 * dim_one),
            nn.Sigmoid()
        )
        self.fc_three = nn.Sequential(
            nn.Linear(many * dim_one, 2 * dim_one),
            nn.Sigmoid()
        )
        self.fc_four = nn.Sequential(
            nn.Linear(many * dim_one, 2 * dim_one),
            nn.Sigmoid()
        )
        self.fc_five = nn.Sequential(
            nn.Linear(many * dim_one, 2 * dim_one),
            nn.Sigmoid()
        )
        self.fc_six = nn.Sequential(
            nn.Linear(many * dim_one, 2 * dim_one),
            nn.Sigmoid()
        )
        self.fc_seven = nn.Sequential(
            nn.Linear(many * dim_one, 2 * dim_one),
            nn.Sigmoid()
        )

        self.conv1_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv5_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv6_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv7_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.conv1_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv2_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv3_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv4_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv5_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))
        self.conv6_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))
        self.conv7_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.gate = nn.Conv2d(many * dim_one, many - 1, kernel_size=1, bias=True)

    def forward(self, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feedback):
        batch, channel, _, _ = feedback.size()

        feat1_ = self.conv1_in(feat1)
        feat2_ = self.conv2_in(feat2)
        feat3_ = self.conv3_in(feat3)
        feat4_ = self.conv4_in(feat4)
        feat5_ = self.conv5_in(feat5)
        feat6_ = self.conv6_in(feat6)
        feat7_ = self.conv7_in(feat7)

        feat1_avg = self.avg_pool(feat1_).view(batch, 1, channel)
        feat2_avg = self.avg_pool(feat2_).view(batch, 1, channel)
        feat3_avg = self.avg_pool(feat3_).view(batch, 1, channel)
        feat4_avg = self.avg_pool(feat4_).view(batch, 1, channel)
        feat5_avg = self.avg_pool(feat5_).view(batch, 1, channel)
        feat6_avg = self.avg_pool(feat6_).view(batch, 1, channel)
        feat7_avg = self.avg_pool(feat7_).view(batch, 1, channel)
        feedback_avg = self.avg_pool(feedback).view(batch, 1, channel)

        combined_fc = torch.cat([feat1_avg, feat2_avg, feat3_avg, feat4_avg,
                                 feat5_avg, feat6_avg, feat7_avg, feedback_avg], dim=1)

        combined_fc_norm = torch.norm(combined_fc, dim=2, keepdim=True)
        combined_fc_norm_t = combined_fc_norm.permute(0, 2, 1)
        combined_fc_t = combined_fc.permute(0, 2, 1)
        mul = torch.bmm(combined_fc, combined_fc_t)
        batch_adj = mul / (combined_fc_norm * combined_fc_norm_t)

        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)

        excitation1 = self.fc_one(feat_cat).view(batch, channel * 2, 1, 1)
        excitation2 = self.fc_two(feat_cat).view(batch, channel * 2, 1, 1)
        excitation3 = self.fc_three(feat_cat).view(batch, channel * 2, 1, 1)
        excitation4 = self.fc_four(feat_cat).view(batch, channel * 2, 1, 1)
        excitation5 = self.fc_five(feat_cat).view(batch, channel * 2, 1, 1)
        excitation6 = self.fc_six(feat_cat).view(batch, channel * 2, 1, 1)
        excitation7 = self.fc_seven(feat_cat).view(batch, channel * 2, 1, 1)

        feedback1 = F.interpolate(feedback, size=feat1_.size()[2:], mode='bilinear')
        feat1_re = torch.cat([feat1_, feedback1], dim=1) * excitation1
        feedback2 = F.interpolate(feedback, size=feat2_.size()[2:], mode='bilinear')
        feat2_re = torch.cat([feat2_, feedback2], dim=1) * excitation2
        feedback3 = F.interpolate(feedback, size=feat3_.size()[2:], mode='bilinear')
        feat3_re = torch.cat([feat3_, feedback3], dim=1) * excitation3
        feedback4 = F.interpolate(feedback, size=feat4_.size()[2:], mode='bilinear')
        feat4_re = torch.cat([feat4_, feedback4], dim=1) * excitation4
        feedback5 = F.interpolate(feedback, size=feat5_.size()[2:], mode='bilinear')
        feat5_re = torch.cat([feat5_, feedback5], dim=1) * excitation5
        feedback6 = F.interpolate(feedback, size=feat6_.size()[2:], mode='bilinear')
        feat6_re = torch.cat([feat6_, feedback6], dim=1) * excitation6
        feedback7 = F.interpolate(feedback, size=feat7_.size()[2:], mode='bilinear')
        feat7_re = torch.cat([feat7_, feedback7], dim=1) * excitation7

        feat1_re = self.conv1_out(feat1_re) + feat1
        feat2_re = self.conv2_out(feat2_re) + feat2
        feat3_re = self.conv3_out(feat3_re) + feat3
        feat4_re = self.conv4_out(feat4_re) + feat4
        feat5_re = self.conv5_out(feat5_re) + feat5
        feat6_re = self.conv6_out(feat6_re) + feat6
        feat7_re = self.conv7_out(feat7_re) + feat7

        return feat1_re, feat2_re, feat3_re, feat4_re, feat5_re, feat6_re, feat7_re


