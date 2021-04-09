#!/usr/bin/python3
#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.ResNet import ResNet34, ResNet50

from module.MMTM import SETriplet2, SEQuart2, SEMany2Many3

class GFM2(nn.Module):
    def __init__(self, GNN=False):
        super(GFM2, self).__init__()
        self.conv1h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv1l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv1f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        # self.gcn_fuse = SEQuart(64, 64, 64, 64)
        self.gcn_fuse2 = SEQuart2(64, 64, 64, 64)
        # self.gcn_fuse3 = SETriplet2(64, 64, 64)
        # self.attention = CAM_Module2(64)
        self.GNN = GNN
    def forward(self, low, high, flow=None, feedback=None):
        if flow is not None:
            if high.size()[2:] != low.size()[2:]:
                high = F.interpolate(high, size=low.size()[2:], mode='bilinear')
            if flow.size()[2:] != low.size()[2:]:
                flow = F.interpolate(flow, size=low.size()[2:], mode='bilinear')

            out1h = self.conv1h(high)
            out2h = self.conv2h(out1h)
            out1l = self.conv1l(low)
            out2l = self.conv2l(out1l)
            out1f = self.conv1f(flow)
            out2f = self.conv2f(out1f)
            if self.GNN:
                fuse = self.gcn_fuse2(out2l, out2h, out2f, feedback)
            else:
                fuse = out2h * out2l * out2f
                # fuse = self.gcn_fuse3(out2l, out2h, out2f)
            out3h = self.conv3h(fuse) + out1h
            out4h = self.conv4h(out3h)
            out3l = self.conv3l(fuse) + out1l
            out4l = self.conv4l(out3l)
            out3f = self.conv3f(fuse) + out1f
            out4f = self.conv4f(out3f)

            return out4l, out4h, out4f
        else:
            if high.size()[2:] != low.size()[2:]:
                high = F.interpolate(high, size=low.size()[2:], mode='bilinear')

            out1h = self.conv1h(high)
            out2h = self.conv2h(out1h)
            out1l = self.conv1l(low)
            out2l = self.conv2l(out1l)
            fuse = out2h * out2l
            out3h = self.conv3h(fuse) + out1h
            out4h = self.conv4h(out3h)
            out3l = self.conv3l(fuse) + out1l
            out4l = self.conv4l(out3l)

            return out4l, out4h

class SFM2(nn.Module):
    def __init__(self):
        super(SFM2, self).__init__()
        self.conv1h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv1l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv1f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.se_triplet = SETriplet2(64, 64, 64)
        # self.attention = CAM_Module(in_dim=64)
    def forward(self, low, high, flow):
        if high.size()[2:] != low.size()[2:]:
            high = F.interpolate(high, size=low.size()[2:], mode='bilinear')
        if flow.size()[2:] != low.size()[2:]:
            flow = F.interpolate(flow, size=low.size()[2:], mode='bilinear')
        out1h = self.conv1h(high)
        out2h = self.conv2h(out1h)
        out1l = self.conv1l(low)
        out2l = self.conv2l(out1l)
        out1f = self.conv1f(flow)
        out2f = self.conv2f(out1f)
        # fuse = out2h * out2l * out2f
        fuse = self.se_triplet(out2h, out2l, out2f)
        out3h = self.conv3h(fuse) + out1h
        out4h = self.conv4h(out3h)
        out3l = self.conv3l(fuse) + out1l
        out4l = self.conv4l(out3l)
        out3f = self.conv3f(fuse) + out1f
        out4f = self.conv4f(out3f)

        return out4l, out4h, out4f

class Decoder_flow(nn.Module):
    def __init__(self):
        super(Decoder_flow, self).__init__()
        self.cfm45  = SFM2()
        self.cfm34  = SFM2()
        self.cfm23  = SFM2()

    def forward(self, out2h, out3h, out4h, out5v, out2f, out3f, out4f, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5

            out4f = F.interpolate(out4f, size=refine4.size()[2:], mode='bilinear')
            out4h, out4v, out4b = self.cfm45(out4h + refine4, out5v, out4f + refine4)
            out4b = F.interpolate(out4b, size=refine3.size()[2:], mode='bilinear')
            out3f = F.interpolate(out3f, size=refine3.size()[2:], mode='bilinear')
            out3h, out3v, out3b = self.cfm34(out3h + refine3, out4v, out3f + out4b + refine3)
            out3b = F.interpolate(out3b, size=refine2.size()[2:], mode='bilinear')
            out2f = F.interpolate(out2f, size=refine2.size()[2:], mode='bilinear')
            out2h, pred, out2b = self.cfm23(out2h+refine2, out3v, out2f + out3b + refine2)
        else:
            out4h, out4v, out4b = self.cfm45(out4h, out5v, out4f)
            out4b = F.interpolate(out4b, size=out3f.size()[2:], mode='bilinear')
            out3h, out3v, out3b = self.cfm34(out3h, out4v, out3f + out4b)
            out3b = F.interpolate(out3b, size=out2f.size()[2:], mode='bilinear')
            out2h, pred, out2b = self.cfm23(out2h, out3v, out2f + out3b)
        return out2h, out3h, out4h, out5v, out2b, out3b, out4b, pred

class Decoder_flow2(nn.Module):
    def __init__(self, GNN=False):
        super(Decoder_flow2, self).__init__()
        self.cfm45  = GFM2(GNN=GNN)
        self.cfm34  = GFM2(GNN=GNN)
        self.cfm23  = GFM2(GNN=GNN)

    def forward(self, out2h, out3h, out4h, out5v, out2f=None, out3f=None, out4f=None, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            if out2f is not None and out3f is not None and out4f is not None:
                out4h, out4v, out4b = self.cfm45(out4h, out5v, out4f, refine4)
                out4b = F.interpolate(out4b, size=out3f.size()[2:], mode='bilinear')
                # out3h, out3v, out3b = self.cfm34(out3h, out4f, out3f + out4b, refine3)
                out3h, out3v, out3b = self.cfm34(out3h, out4v, out3f + out4b, refine3)
                out3b = F.interpolate(out3b, size=out2f.size()[2:], mode='bilinear')
                out2h, pred, out2b = self.cfm23(out2h, out3v, out2f + out3b, refine2)
            else:
                out4h, out4v = self.cfm45(out4h + refine4, out5v)
                out3h, out3v = self.cfm34(out3h + refine3, out4v)
                out2h, pred = self.cfm23(out2h + refine2, out3v)
        else:
            if out2f is not None and out3f is not None and out4f is not None:
                out4h, out4v, out4b = self.cfm45(out4h, out5v, out4f)
                out4b = F.interpolate(out4b, size=out3f.size()[2:], mode='bilinear')
                out3h, out3v, out3b = self.cfm34(out3h, out4v, out3f + out4b)
                out3b = F.interpolate(out3b, size=out2f.size()[2:], mode='bilinear')
                out2h, pred, out2b = self.cfm23(out2h, out3v, out2f + out3b)
            else:
                out4h, out4v = self.cfm45(out4h, out5v)
                out3h, out3v = self.cfm34(out3h, out4v)
                out2h, pred = self.cfm23(out2h, out3v)
        if out2f is not None and out3f is not None and out4f is not None:
            return out2h, out3h, out4h, out5v, out2b, out3b, out4b, pred
        else:
            return out2h, out3h, out4h, out5v, pred

class INet(nn.Module):
    def __init__(self, cfg, GNN=False):
        super(INet, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet50(nInputChannels=3, os=32, pretrained=False)
        self.flow_bkbone = ResNet34(nInputChannels=3, os=16, pretrained=False)
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.flow_align4 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.flow_align3 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.flow_align2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.flow_align1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder1 = Decoder_flow()
        self.decoder2 = Decoder_flow2(GNN=GNN)
        self.decoder3 = Decoder_flow2(GNN=GNN)
        self.se_many = SEMany2Many3(5, 4, 64)
        # self.se_many2 = SEMany2Many4(6, 64)
        # self.gnn_embedding = GNN_Embedding()
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # self.linearf1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearf2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearf3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearf4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, flow=None, shape=None):
        out5v, out2h, _, out3h, out4h = self.bkbone(x) # layer1, layer2, layer3, layer4
        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)

        flow_layer4, flow_layer1, _, flow_layer2, flow_layer3 = self.flow_bkbone(flow)
        out1f, out2f = self.flow_align1(flow_layer1), self.flow_align2(flow_layer2)
        out3f, out4f = self.flow_align3(flow_layer3), self.flow_align4(flow_layer4)
        out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred1 = self.decoder1(out2h, out3h, out4h, out5v, out2f, out3f, out4f)
        out2f_scale, out3f_scale, out4f_scale = out2f.size()[2:], out3f.size()[2:], out4f.size()[2:]
        # out2f, out3f, out4f = self.se_many_flow(feat_flow_list, pred1)
        out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred2 = self.decoder2(out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred1)
        # feat_list2 = [out2h, out3h, out4h, out5v, out4f]
        out2h, out3h, out4h, out5v = self.se_many(out2h, out3h, out4h, out5v, pred2)
        # out2h, out3h, out4h, out5v, out2f, out3f, out4f = self.se_many(out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred2)
        # out2h, out3h, out4h, out5v, out4f = self.se_many2(feat_list2, pred2)
        out2f = F.interpolate(out2f, size=out2f_scale, mode='bilinear')
        out3f = F.interpolate(out3f, size=out3f_scale, mode='bilinear')
        out4f = F.interpolate(out4f, size=out4f_scale, mode='bilinear')
        out2h, out3h, out4h, out5v, out1f, out3f, out4f, pred3 = self.decoder3(out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred2)

        shape = x.size()[2:] if shape is None else shape

        pred1a = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2a = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')
        pred3a = F.interpolate(self.linearp3(pred3), size=shape, mode='bilinear')

        out2h_p = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h_p = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h_p = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h_p = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')

        out2f_p = F.interpolate(self.linearf2(out2f), size=shape, mode='bilinear')
        out3f_p = F.interpolate(self.linearf3(out3f), size=shape, mode='bilinear')
        out4f_p = F.interpolate(self.linearf4(out4f), size=shape, mode='bilinear')

        return pred1a, pred2a, pred3a, out2h_p, out3h_p, out4h_p, out5h_p, out2h, out3h, out4h, out5v,\
               out2f_p, out3f_p, out4f_p, out2f, out3f, out4f


if __name__ == '__main__':
        net = INet(cfg=None, GNN=True)
        input = torch.zeros([2, 3, 380, 380])
        output = net(input, input)
