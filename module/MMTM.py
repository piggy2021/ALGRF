import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from module.GCN import GCN
import math
import torch.nn.functional as F

coarse_adj_list = [
            # 1  2  3
            [0.333, 0.333, 0.333],  # 1
            [0.333, 0.333, 0.333],  # 2
            [0.333, 0.333, 0.333],  # 3
        ]

coarse_adj_list2 = [
            # 1  2  3  4
            [0.25, 0.25, 0.25, 0.25],  # 1
            [0.25, 0.25, 0.25, 0.25],  # 2
            [0.25, 0.25, 0.25, 0.25],  # 3
            [0.25, 0.25, 0.25, 0.25],  # 4
        ]

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

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, GCN):
            pass
        else:
            m.initialize()

class MMTM(nn.Module):
    def __init__(self, dim_a, dim_b, ratio):
        super(MMTM, self).__init__()
        dim = dim_a + dim_b
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_a = nn.Linear(dim_out, dim_a)
        self.fc_b = nn.Linear(dim_out, dim_b)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def initialize(self):
        nn.init.kaiming_normal_(self.fc_squeeze.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_a.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_b.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, a, b):
        squeeze_array = []
        for tensor in [a, b]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_a(excitation)
        sk_out = self.fc_b(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(a.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(b.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return a * vis_out, b * sk_out

class SETriplet(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, dim_out):
        super(SETriplet, self).__init__()
        dim = dim_a + dim_b + dim_c
        # self.fc_squeeze = nn.Linear(dim, dim_out)
        self.fc_one = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_a),
            nn.Sigmoid()
        )
        self.fc_two = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_b),
            nn.Sigmoid()
        )
        self.fc_three = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_b),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.gate_a = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.gate_b = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.gate_c = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def initialize(self):
        weight_init(self)

    def forward(self, a, b, c):
        batch, channel, _, _ = a.size()
        combined = torch.cat([a, b, c], dim=1)
        combined_fc = self.avg_pool(combined).view(batch, channel * 3)
        excitation1 = self.fc_one(combined_fc).view(batch, channel, 1, 1)
        excitation2 = self.fc_two(combined_fc).view(batch, channel, 1, 1)
        excitation3 = self.fc_three(combined_fc).view(batch, channel, 1, 1)

        weighted_feat_a = a + excitation2 * b + excitation3 * c
        weighted_feat_b = b + excitation1 * a + excitation3 * c
        weighted_feat_c = c + excitation1 * a + excitation2 * b

        feat_cat = torch.cat([weighted_feat_a, weighted_feat_b, weighted_feat_c], dim=1)
        atten_a = self.gate_a(feat_cat)
        atten_b = self.gate_b(feat_cat)
        atten_c = self.gate_c(feat_cat)

        attention_vector = torch.cat([atten_a, atten_b, atten_c], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_a, attention_vector_b, attention_vector_c = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :], attention_vector[:, 2:3, :, :]

        merge_feature = a * attention_vector_a + b * attention_vector_b + c * attention_vector_c
        out_a = torch.relu((a + merge_feature) / 2)
        out_b = torch.relu((b + merge_feature) / 2)
        out_c = torch.relu((c + merge_feature) / 2)
        return out_a, out_b, out_c, merge_feature

class SETriplet2(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c):
        super(SETriplet2, self).__init__()
        dim = dim_a + dim_b + dim_c

        self.gcn = GCN(3, 64, 64)
        coarse_adj = np.ones([3, 3])
        self.adj = torch.from_numpy(L_Matrix(coarse_adj, 3)).float()

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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        # self.gate_a = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_b = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_c = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(dim, 64, kernel_size=1, bias=True)

    def initialize(self):
        weight_init(self)

    def forward(self, a, b, c):
        batch, channel, _, _ = a.size()
        combined = torch.cat([a, b, c], dim=1)
        combined_fc = self.avg_pool(combined).view(batch, 3, channel)
        # batch_adj = self.adj.repeat(batch, 1, 1)
        # batch_adj = batch_adj.cuda(device_id)

        combined_fc_norm = torch.norm(combined_fc, dim=2, keepdim=True)
        combined_fc_norm_t = combined_fc_norm.permute(0, 2, 1)
        combined_fc_t = combined_fc.permute(0, 2, 1)
        mul = torch.bmm(combined_fc, combined_fc_t)
        batch_adj = mul / (combined_fc_norm * combined_fc_norm_t)
        # batch_adj_norm = torch.norm(batch_adj, dim=2, keepdim=True)
        # batch_adj = batch_adj / batch_adj_norm

        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)

        excitation1 = self.fc_one(feat_cat).view(batch, channel, 1, 1)
        excitation2 = self.fc_two(feat_cat).view(batch, channel, 1, 1)
        excitation3 = self.fc_three(feat_cat).view(batch, channel, 1, 1)

        weighted_feat_a = a + excitation2 * b + excitation3 * c
        weighted_feat_b = b + excitation1 * a + excitation3 * c
        weighted_feat_c = c + excitation1 * a + excitation2 * b
        feat_cat = torch.cat([weighted_feat_a, weighted_feat_b, weighted_feat_c], dim=1)

        # merge = excitation1 * a + excitation2 * b + excitation3 * c
        # feat_cat = torch.cat([a + merge, b + merge, c + merge], dim=1)
        merge_feature = self.gate(feat_cat)
        # atten_a = self.gate_a(feat_cat)
        # atten_b = self.gate_b(feat_cat)
        # atten_c = self.gate_c(feat_cat)

        # attention_vector = torch.cat([atten_a, atten_b, atten_c], dim=1)
        # attention_vector = self.softmax(attention_vector)
        # attention_vector_a, attention_vector_b, attention_vector_c = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :], attention_vector[:, 2:3, :, :]
        #
        # merge_feature = a * attention_vector_a + b * attention_vector_b + c * attention_vector_c
        # out_a = torch.relu((a + merge_feature) / 2)
        # out_b = torch.relu((b + merge_feature) / 2)
        # out_c = torch.relu((c + merge_feature) / 2)
        return merge_feature

class SEQuart(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, dim_d):
        super(SEQuart, self).__init__()
        dim = dim_a + dim_b + dim_c + dim_d

        self.gcn = GCN(4, 64, 64)
        coarse_adj = np.ones([4, 4])
        self.adj = torch.from_numpy(L_Matrix(coarse_adj, 4)).float()

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

    def initialize(self):
        weight_init(self)

    def forward(self, low, high, flow, feedback):
        batch, channel, _, _ = low.size()
        combined = torch.cat([low, high, flow, feedback], dim=1)
        combined_fc = self.avg_pool(combined).view(batch, 4, channel)
        # batch_adj = self.adj.repeat(batch, 1, 1)
        # batch_adj = batch_adj.cuda(device_id)

        combined_fc_norm = torch.norm(combined_fc, dim=2, keepdim=True)
        combined_fc_norm_t = combined_fc_norm.permute(0, 2, 1)
        combined_fc_t = combined_fc.permute(0, 2, 1)
        mul = torch.bmm(combined_fc, combined_fc_t)
        batch_adj = mul / (combined_fc_norm * combined_fc_norm_t)
        # batch_adj_norm = torch.norm(batch_adj, dim=2, keepdim=True)
        # batch_adj = batch_adj / batch_adj_norm

        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)
        # feat_cat = self.avg_pool(feat_cat).view(batch, 4 * channel)
        excitation1 = self.fc_one(feat_cat).view(batch, channel, 1, 1)
        excitation2 = self.fc_two(feat_cat).view(batch, channel, 1, 1)
        excitation3 = self.fc_three(feat_cat).view(batch, channel, 1, 1)
        excitation4 = self.fc_four(feat_cat).view(batch, channel, 1, 1)

        weighted_feat_a = low + excitation2 * high + excitation3 * flow + excitation4 * feedback
        weighted_feat_b = excitation1 * low + high + excitation3 * flow + excitation4 * feedback
        weighted_feat_c = excitation1 * low + excitation2 * high + flow + excitation4 * feedback
        weighted_feat_d = excitation1 * low + excitation2 * high + excitation3 * flow + feedback
        feat_cat = torch.cat([weighted_feat_a, weighted_feat_b, weighted_feat_c, weighted_feat_d], dim=1)

        # merge = excitation1 * low + excitation2 * high + excitation3 * flow + excitation4 * feedback
        # feat_cat = torch.cat([low + merge, high + merge, flow + merge, feedback + merge], dim=1)
        # atten_a = self.gate_a(feat_cat)
        # atten_b = self.gate_b(feat_cat)
        # atten_c = self.gate_c(feat_cat)
        # atten_d = self.gate_d(feat_cat)
        # attention_vector = torch.cat([atten_a, atten_b, atten_c, atten_d], dim=1)

        attention_vector = self.gate(feat_cat)
        # attention_vector = self.softmax(attention_vector)
        #
        # attention_vector_a, attention_vector_b = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        # attention_vector_c, attention_vector_d = attention_vector[:, 2:3, :, :], attention_vector[:, 3:4, :, :]
        # merge_feature = low * attention_vector_a + high * attention_vector_b + \
        #                 flow * attention_vector_c + feedback * attention_vector_d
        # bug backup
        # merge_feature = low * attention_vector_a + high * attention_vector_b + \
        #                 flow * attention_vector_c * feedback * attention_vector_d

        return attention_vector

class SEQuart2(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, dim_d):
        super(SEQuart2, self).__init__()
        dim = dim_a + dim_b + dim_c + dim_d

        # self.gcn = GCN(4, 64, 64)
        # coarse_adj = np.ones([4, 4])
        # self.adj = torch.from_numpy(L_Matrix(coarse_adj, 4)).float()

        # self.fc_one = nn.Sequential(
        #     nn.Linear(dim, dim_a),
        #     nn.Sigmoid()
        # )
        # self.fc_two = nn.Sequential(
        #     nn.Linear(dim, dim_b),
        #     nn.Sigmoid()
        # )
        # self.fc_three = nn.Sequential(
        #     nn.Linear(dim, dim_c),
        #     nn.Sigmoid()
        # )
        # self.fc_four = nn.Sequential(
        #     nn.Linear(dim, dim_c),
        #     nn.Sigmoid()
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        # self.gate_a = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_b = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_c = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_d = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(dim, 64, kernel_size=1, bias=True)

    def initialize(self):
        weight_init(self)

    def forward(self, low, high, flow, feedback):
        batch, channel, _, _ = low.size()
        combined = torch.cat([low, high, flow, feedback], dim=1)


        # merge = excitation1 * low + excitation2 * high + excitation3 * flow + excitation4 * feedback
        # feat_cat = torch.cat([low + merge, high + merge, flow + merge, feedback + merge], dim=1)
        # atten_a = self.gate_a(feat_cat)
        # atten_b = self.gate_b(feat_cat)
        # atten_c = self.gate_c(feat_cat)
        # atten_d = self.gate_d(feat_cat)
        # attention_vector = torch.cat([atten_a, atten_b, atten_c, atten_d], dim=1)

        attention_vector = self.gate(combined)
        # attention_vector = self.softmax(attention_vector)
        #
        # attention_vector_a, attention_vector_b = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        # attention_vector_c, attention_vector_d = attention_vector[:, 2:3, :, :], attention_vector[:, 3:4, :, :]
        # merge_feature = low * attention_vector_a + high * attention_vector_b + \
        #                 flow * attention_vector_c + feedback * attention_vector_d
        # bug backup
        # merge_feature = low * attention_vector_a + high * attention_vector_b + \
        #                 flow * attention_vector_c * feedback * attention_vector_d

        return attention_vector

class SEMany2Many(nn.Module):
    def __init__(self, many, dim_one):
        super(SEMany2Many, self).__init__()


        self.gcn = GCN(many, dim_one, dim_one)
        coarse_adj = np.ones([many, many])
        self.adj = torch.from_numpy(L_Matrix(coarse_adj, many)).float()
        self.fc_one = nn.Sequential(
            nn.Linear(many * dim_one, dim_one),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.softmax = nn.Softmax(dim=1)

        # self.gate_a = nn.Conv2d(dim_one, 1, kernel_size=1, bias=True)
        # self.gate_b = nn.Conv2d(dim_one, 1, kernel_size=1, bias=True)
        # self.gate_c = nn.Conv2d(dim_one, 1, kernel_size=1, bias=True)
        # self.gate_d = nn.Conv2d(dim_one, 1, kernel_size=1, bias=True)
        # self.gate_e = nn.Conv2d(dim_one, 1, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(dim_one, 1, kernel_size=1, bias=True)

    def initialize(self):
        weight_init(self)

    def forward(self, feat_list, feedback):
        batch, channel, _, _ = feedback.size()
        # combined = torch.cat([low, high, flow, feedback], dim=1)
        # feat1_avg = self.avg_pool(feat1).view(batch, 1, channel)
        # feat2_avg = self.avg_pool(feat2).view(batch, 1, channel)
        # feat3_avg = self.avg_pool(feat3).view(batch, 1, channel)
        # feat4_avg = self.avg_pool(feat4).view(batch, 1, channel)
        feat_avg = []
        for feat in feat_list:
            feat_avg.append(self.avg_pool(feat).view(batch, 1, channel))
        feedback_avg = self.avg_pool(feedback).view(batch, 1, channel)
        feat_avg.append(feedback_avg)
        combined_fc = torch.cat(feat_avg, dim=1)
        # combined_fc = self.avg_pool(combined).view(batch, 4, channel)
        batch_adj = self.adj.repeat(batch, 1, 1)
        batch_adj = batch_adj.cuda(device_id)
        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)

        excitation = self.fc_one(feat_cat).view(batch, channel, 1, 1)
        gate = torch.sigmoid(self.gate(feedback * excitation))
        feat_output = []
        for feat in feat_list:
            gate = F.interpolate(gate, size=feat.size()[2:], mode='bilinear')
            feat_output.append(feat * gate + feat)

        return feat_output

class SEMany2Many2(nn.Module):
    def __init__(self, many, dim_one):
        super(SEMany2Many2, self).__init__()

        self.gcn = GCN(many, dim_one, dim_one)
        coarse_adj = np.ones([many, many])
        self.adj = torch.from_numpy(L_Matrix(coarse_adj, many)).float()
        self.fc_one = nn.Sequential(
            nn.Linear(many * dim_one, many * dim_one),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.gate = nn.Conv2d(many * dim_one, many - 1, kernel_size=1, bias=True)

    def initialize(self):
        weight_init(self)

    def forward(self, feat_list, feedback):
        batch, channel, _, _ = feedback.size()

        feat_align = []
        feat_align.append(F.interpolate(feedback, size=(48, 48), mode='bilinear'))
        for feat in feat_list:
            feat_resized = F.interpolate(feat, size=(48, 48), mode='bilinear')
            feat_align.append(feat_resized)

        feat_avg = []
        for feat in feat_list:
            feat_avg.append(self.avg_pool(feat).view(batch, 1, channel))
        feedback_avg = self.avg_pool(feedback).view(batch, 1, channel)
        feat_avg.append(feedback_avg)
        combined_fc = torch.cat(feat_avg, dim=1)
        # combined_fc = self.avg_pool(combined).view(batch, 4, channel)
        batch_adj = self.adj.repeat(batch, 1, 1)
        batch_adj = batch_adj.cuda(device_id)
        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)

        excitation = self.fc_one(feat_cat).view(batch, channel * (len(feat_list) + 1), 1, 1)
        feat_align = torch.cat(feat_align, dim=1)
        gate = self.softmax(self.gate(feat_align * excitation))
        feat_output = []
        for i, feat in enumerate(feat_list):
            gate_ = F.interpolate(gate[:, i:i+1, :, :], size=feat.size()[2:], mode='bilinear')
            feat_output.append(feat * gate_ + feat)

        return feat_output

class SEMany2Many3(nn.Module):
    def __init__(self, many, many2, dim_one):
        super(SEMany2Many3, self).__init__()

        self.gcn = GCN(many, dim_one, dim_one)
        coarse_adj = np.ones([many, many])
        self.adj = torch.from_numpy(L_Matrix(coarse_adj, many)).float()

        # self.gcn2 = GCN(many2, dim_one, dim_one)
        # coarse_adj2 = np.ones([many2, many2])
        # self.adj2 = torch.from_numpy(L_Matrix(coarse_adj2, many2)).float()

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
        # self.fc_five = nn.Sequential(
        #     nn.Linear(many * dim_one, 2 * dim_one),
        #     nn.Sigmoid()
        # )
        # self.fc_six = nn.Sequential(
        #     nn.Linear(many * dim_one, 2 * dim_one),
        #     nn.Sigmoid()
        # )
        # self.fc_seven = nn.Sequential(
        #     nn.Linear(many * dim_one, 2 * dim_one),
        #     nn.Sigmoid()
        # )

        self.conv1_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.conv5_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        #                               nn.ReLU(inplace=True))
        # self.conv6_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        #                               nn.ReLU(inplace=True))
        # self.conv7_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        #                               nn.ReLU(inplace=True))

        self.conv1_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv2_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv3_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv4_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        # self.conv5_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        #                                nn.ReLU(inplace=True))
        # self.conv6_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        #                                nn.ReLU(inplace=True))
        # self.conv7_out = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        #                                nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.gate = nn.Conv2d(many * dim_one, many - 1, kernel_size=1, bias=True)
        # self.gate2 = nn.Conv2d(many2 * dim_one, many2 - 1, kernel_size=1, bias=True)

    def initialize(self):
        weight_init(self)

    def forward(self, feat1, feat2, feat3, feat4, feedback):
        batch, channel, _, _ = feedback.size()

        feat1_ = self.conv1_in(feat1)
        feat2_ = self.conv2_in(feat2)
        feat3_ = self.conv3_in(feat3)
        feat4_ = self.conv4_in(feat4)
        # feat5_ = self.conv5_in(feat5)
        # feat6_ = self.conv6_in(feat6)
        # feat7_ = self.conv7_in(feat7)

        feat1_avg = self.avg_pool(feat1_).view(batch, 1, channel)
        feat2_avg = self.avg_pool(feat2_).view(batch, 1, channel)
        feat3_avg = self.avg_pool(feat3_).view(batch, 1, channel)
        feat4_avg = self.avg_pool(feat4_).view(batch, 1, channel)
        # feat5_avg = self.avg_pool(feat5_).view(batch, 1, channel)
        # feat6_avg = self.avg_pool(feat6_).view(batch, 1, channel)
        # feat7_avg = self.avg_pool(feat7_).view(batch, 1, channel)
        feedback_avg = self.avg_pool(feedback).view(batch, 1, channel)

        combined_fc = torch.cat([feat1_avg, feat2_avg, feat3_avg, feat4_avg, feedback_avg], dim=1)
        # combined_fc2 = torch.cat([feat5_avg, feat6_avg, feat7_avg, feedback_avg], dim=1)
        # combined_fc = self.avg_pool(combined).view(batch, 4, channel)
        # batch_adj = self.adj.repeat(batch, 1, 1)
        # batch_adj = batch_adj.cuda(device_id)

        combined_fc_norm = torch.norm(combined_fc, dim=2, keepdim=True)
        combined_fc_norm_t = combined_fc_norm.permute(0, 2, 1)
        combined_fc_t = combined_fc.permute(0, 2, 1)
        mul = torch.bmm(combined_fc, combined_fc_t)
        batch_adj = mul / (combined_fc_norm * combined_fc_norm_t)
        # batch_adj_norm = torch.norm(batch_adj, dim=2, keepdim=True)
        # batch_adj = batch_adj / batch_adj_norm

        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)

        # batch_adj2 = self.adj2.repeat(batch, 1, 1)
        # batch_adj2 = batch_adj2.cuda(device_id)
        # feat_mean2, feat_cat2 = self.gcn2(combined_fc2, batch_adj2)
        # feat_cat = combined_fc.view(batch, 5 * channel)
        excitation1 = self.fc_one(feat_cat).view(batch, channel * 2, 1, 1)
        excitation2 = self.fc_two(feat_cat).view(batch, channel * 2, 1, 1)
        excitation3 = self.fc_three(feat_cat).view(batch, channel * 2, 1, 1)
        excitation4 = self.fc_four(feat_cat).view(batch, channel * 2, 1, 1)
        # excitation5 = self.fc_five(feat_cat).view(batch, channel * 2, 1, 1)
        # excitation6 = self.fc_six(feat_cat).view(batch, channel * 2, 1, 1)
        # excitation7 = self.fc_seven(feat_cat).view(batch, channel * 2, 1, 1)

        feedback1 = F.interpolate(feedback, size=feat1_.size()[2:], mode='bilinear')
        feat1_re = torch.cat([feat1_, feedback1], dim=1) * excitation1
        feedback2 = F.interpolate(feedback, size=feat2_.size()[2:], mode='bilinear')
        feat2_re = torch.cat([feat2_, feedback2], dim=1) * excitation2
        feedback3 = F.interpolate(feedback, size=feat3_.size()[2:], mode='bilinear')
        feat3_re = torch.cat([feat3_, feedback3], dim=1) * excitation3
        feedback4 = F.interpolate(feedback, size=feat4_.size()[2:], mode='bilinear')
        feat4_re = torch.cat([feat4_, feedback4], dim=1) * excitation4
        # feedback5 = F.interpolate(feedback, size=feat5_.size()[2:], mode='bilinear')
        # feat5_re = torch.cat([feat5_, feedback5], dim=1) * excitation5
        # feedback6 = F.interpolate(feedback, size=feat6_.size()[2:], mode='bilinear')
        # feat6_re = torch.cat([feat6_, feedback6], dim=1) * excitation6
        # feedback7 = F.interpolate(feedback, size=feat7_.size()[2:], mode='bilinear')
        # feat7_re = torch.cat([feat7_, feedback7], dim=1) * excitation7

        feat1_re = self.conv1_out(feat1_re) + feat1
        feat2_re = self.conv2_out(feat2_re) + feat2
        feat3_re = self.conv3_out(feat3_re) + feat3
        feat4_re = self.conv4_out(feat4_re) + feat4
        # feat5_re = self.conv5_out(feat5_re) + feat5
        # feat6_re = self.conv6_out(feat6_re) + feat6
        # feat7_re = self.conv7_out(feat7_re) + feat7

        return feat1_re, feat2_re, feat3_re, feat4_re

class SEMany2Many4(nn.Module):
    def __init__(self, many, dim_one):
        super(SEMany2Many4, self).__init__()

        self.conv1_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4_in = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv1_out = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv2_out = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv3_out = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv4_out = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

    def initialize(self):
        weight_init(self)

    def forward(self, feat1, feat2, feat3, feat4, feedback):

        feat1_ = self.conv1_in(feat1)
        feat2_ = self.conv2_in(feat2)
        feat3_ = self.conv3_in(feat3)
        feat4_ = self.conv4_in(feat4)

        feedback1 = F.interpolate(feedback, size=feat1_.size()[2:], mode='bilinear')
        feedback2 = F.interpolate(feedback, size=feat2_.size()[2:], mode='bilinear')
        feedback3 = F.interpolate(feedback, size=feat3_.size()[2:], mode='bilinear')
        feedback4 = F.interpolate(feedback, size=feat4_.size()[2:], mode='bilinear')

        feat1_re = self.conv1_out(feat1_ + feedback1) + feat1
        feat2_re = self.conv2_out(feat2_ + feedback2) + feat2
        feat3_re = self.conv3_out(feat3_ + feedback3) + feat3
        feat4_re = self.conv4_out(feat4_ + feedback4) + feat4

        return feat1_re, feat2_re, feat3_re, feat4_re

if __name__ == '__main__':
        input = torch.rand([2, 64, 24, 24])
        # net = SEQuart(64, 64, 64, 64)
        feat_list = [input, input, input, input, input]
        # net = SEMany2Many3(5, 4, 64)
        # net = SEQuart(64, 64, 64, 64)
        # output = net(input, input, input, input)
        input2 = torch.rand(2, 4, 64)
        input2_norm = torch.norm(input2, dim=2, keepdim=True)
        input2_norm_t = input2_norm.permute(0, 2, 1)
        input2_t = input2.permute(0, 2, 1)
        mul = torch.bmm(input2, input2_t)
        print(mul.shape)
        a = mul / (input2_norm * input2_norm_t)
        print(a)
        b = torch.norm(a, dim=2, keepdim=True)
        c = a / b
        print(c)
        # output = torch.cosine_similarity(input2, input2, dim=2)
        # print(output.shape)
        # print(output)

