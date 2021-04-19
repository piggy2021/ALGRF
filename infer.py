import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import davis_path, visal_path, davsod_path
from models.net import INet
from utils_mine import MaxMinNormalization, check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from matplotlib import pyplot as plt


torch.manual_seed(2020)

# set which gpu to use
device_id = 0
torch.cuda.set_device(device_id)
ckpt_path = './pretrained'

args = {
    'gnn': True,
    'save_results': True,  # whether to save the resulting masks
    'input_size': (380, 380)
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

# to_test = {'davis': os.path.join(davis_path, 'davis_test2')}
# gt_root = os.path.join(davis_path, 'GT')
# flow_root = os.path.join(davis_path, 'flow')
# imgs_path = os.path.join(davis_path, 'davis_test2_single.txt')

to_test = {'ViSal': os.path.join(visal_path, 'ViSal_test')}
gt_root = os.path.join(visal_path, 'GT')
flow_root = os.path.join(visal_path, 'flow')
imgs_path = os.path.join(visal_path, 'ViSal_test_single.txt')

# to_test = {'VOS': os.path.join(vos_path, 'VOS_test')}
# gt_root = os.path.join(vos_path, 'GT')
# flow_root = os.path.join(vos_path, 'flow')
# imgs_path = os.path.join(vos_path, 'VOS_test_single.txt')

# to_test = {'DAVSOD': os.path.join(davsod_path, 'DAVSOD_test')}
# gt_root = os.path.join(davsod_path, 'GT')
# flow_root = os.path.join(davsod_path, 'flow')
# imgs_path = os.path.join(davsod_path, 'DAVSOD_test_single.txt')


def main():

    net = INet(cfg=None, GNN=args['gnn'])

    print ('load snapshot for testing')
    net.load_state_dict(torch.load(os.path.join(ckpt_path, 'ALGRF_pretrained'),
                               map_location='cuda:' + str(device_id)))
    net.eval()
    net.cuda()
    results = {}

    with torch.no_grad():

        for name, root in to_test.items():

            precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()

            if args['save_results']:
                check_mkdir(os.path.join(ckpt_path, '%s' % name))
            img_list = [i_id.strip() for i_id in open(imgs_path)]
            video = ''
            pre_predict = None
            for idx, img_name in enumerate(img_list):
                print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                print(img_name)
                if video != img_name.split('/')[0]:
                    video = img_name.split('/')[0]
                    if name != 'VOS':
                        continue
                    if name == 'VOS' or name == 'DAVSOD':
                        img = Image.open(os.path.join(root, img_name + '.png')).convert('RGB')
                    else:
                        img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
                    flow = Image.open(os.path.join(flow_root, img_name + '.jpg')).convert('RGB')
                    shape = img.size
                    img = img.resize(args['input_size'])
                    flow = flow.resize(args['input_size'])
                    img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
                    flow_var = Variable(img_transform(flow).unsqueeze(0), volatile=True).cuda()

                    prediction, prediction2, prediction3 = net(img_var, flow_var)
                    prediction = torch.sigmoid(prediction3)
                else:
                    if name == 'VOS' or name == 'DAVSOD':
                        img = Image.open(os.path.join(root, img_name + '.png')).convert('RGB')
                    else:
                        img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
                    if name == 'davis':
                        flow = Image.open(os.path.join(flow_root, img_name + '.jpg')).convert('RGB')
                    else:
                        flow = Image.open(os.path.join(flow_root, img_name + '.jpg')).convert('RGB')
                    # flow = Image.open(os.path.join(flow_root, img_name + '.jpg')).convert('RGB')
                    shape = img.size
                    img = img.resize(args['input_size'])
                    flow = flow.resize(args['input_size'])
                    img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
                    flow_var = Variable(img_transform(flow).unsqueeze(0), volatile=True).cuda()

                    prediction, prediction2, prediction3 = net(img_var, flow_var)
                    prediction = torch.sigmoid(prediction3)

                precision = to_pil(prediction.data.squeeze(0).cpu())
                precision = precision.resize(shape)
                prediction = np.array(precision)
                prediction = prediction.astype('float')

                prediction = MaxMinNormalization(prediction, prediction.max(), prediction.min()) * 255.0
                prediction = prediction.astype('uint8')

                gt = np.array(Image.open(os.path.join(gt_root, img_name + '.png')).convert('L'))
                precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    precision_record[pidx].update(p)
                    recall_record[pidx].update(r)
                mae_record.update(mae)

                if args['save_results']:
                    folder, sub_name = os.path.split(img_name)
                    save_path = os.path.join(ckpt_path, '%s' % name, folder)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    Image.fromarray(prediction).save(os.path.join(save_path, sub_name + '.png'))

            fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

            results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

    print ('test results:')
    print (results)
if __name__ == '__main__':
    main()

