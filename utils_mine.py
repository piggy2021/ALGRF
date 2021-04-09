import torch
import numpy as np
from matplotlib import pyplot as plt
import os

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x


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

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def cal_precision_recall_mae(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure
