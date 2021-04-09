import torch
from models.net import SNet
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('/home/tangyi/code/SVS')
plt.style.use('classic')

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x

def fuse_MGA_F3Net(mga_model_path, f3net_path, net, device_id=0):
    # net = SNet(cfg=None).cuda()
    f3_model = torch.load(f3net_path, map_location='cuda:' + str(device_id))
    mga_model = torch.load(mga_model_path, map_location='cuda:' + str(device_id))
    mga_keys = list(mga_model.keys())
    flow_keys = [key for key in mga_keys if key.find('resnet_aspp.backbone_features') > -1]
    m_dict = net.state_dict()
    for k in m_dict.keys():
        if k in f3_model.keys():
            print('loading F3Net key:', k)
            param = f3_model.get(k)
            m_dict[k].data = param
        elif k.find('flow_bkbone') > -1:
            print('loading MGA key:', k)
            k_tmp = k.replace('flow_bkbone', 'resnet_aspp.backbone_features')
            # k_tmp.replace('flow_bkbone', 'resnet_aspp.backbone_features')
            m_dict[k].data = mga_model.get(k_tmp)
        else:
            print('not loading key:', k)

    net.load_state_dict(m_dict)
    return net


def load_MGA(mga_model, model_path, device_id=0):
    pretrain_weights = torch.load(model_path, map_location='cuda:' + str(device_id))
    pretrain_keys = list(pretrain_weights.keys())
    print(pretrain_keys)
    pretrain_keys = [key for key in pretrain_keys if not key.endswith('num_batches_tracked')]
    net_keys = list(mga_model.state_dict().keys())

    for key in net_keys:
        # key_ = 'module.' + key
        key_ = key
        if key_ in pretrain_keys:
            assert (mga_model.state_dict()[key].size() == pretrain_weights[key_].size())
            mga_model.state_dict()[key].copy_(pretrain_weights[key_])
        else:
            print('missing key: ', key_)
    print('loaded pre-trained weights.')
    return mga_model

def load_part_of_model(new_model, src_model_path, device_id=0):
    src_model = torch.load(src_model_path, map_location='cuda:' + str(device_id))
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        if k in m_dict.keys():
            param = src_model.get(k)
            m_dict[k].data = param
            print('loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model2(new_model, src_model_path, device_id=0):
    src_model = torch.load(src_model_path, map_location='cuda:' + str(device_id))
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print(k)
        param = src_model.get(k)
        k = k.replace('module.', '')
        m_dict[k].data = param

    new_model.load_state_dict(m_dict)
    return new_model

def visualize(input, save_path):
    input = input.data.cpu().numpy()
    for i in range(input.shape[1]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(input[0, i, :, :])

    plt.savefig(save_path)

def visualize_vec(input_vec, save_path):
    input = input_vec.data.cpu().numpy()
    input = np.squeeze(input)
    input = np.tile(input, (64, 1))
    plt.imshow(input)
    plt.colorbar()
    plt.savefig(save_path)

if __name__ == '__main__':
    ckpt_path = './ckpt'
    exp_name = 'VideoSaliency_2019-05-14 17:13:16'

    args = {
        'snapshot': '30000',  # your snapshot filename (exclude extension name)
        'crf_refine': False,  # whether to use crf to refine results
        'save_results': True,  # whether to save the resulting masks
        'input_size': (473, 473)
    }
    a = torch.rand([1, 64, 1, 1])
    # visualize_vec(a, 'a.png')
    # from MGA.mga_model import MGA_Network
    # a = MGA_Network(nInputChannels=3, n_classes=1, os=16,
    #             img_backbone_type='resnet101', flow_backbone_type='resnet34')
    # load_MGA(a, '../pre-trained/MGA_trained.pth')

    net = SNet(cfg=None).cuda()
    net = fuse_MGA_F3Net('../pre-trained/MGA_trained.pth', '../pre-trained/F3Net.pth', net)
    torch.save(net.state_dict(), '../pre-trained/SNet.pth')
    # net = load_part_of_model(net, '../pre-trained/SNet.pth')
    # input = torch.zeros([2, 3, 380, 380]).cuda()
    # output = net(input, input)
    # print(output[0].size())
    # src_model_path = os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')
    # net = R3Net(motion='GRU')
    # net = load_part_of_model(net, src_model_path)
