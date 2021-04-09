import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, \
    davis_path, fbms_path, mcl_path, uvsd_path, visal_path, vos_path, segtrack_path, davsod_path
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
# from models.net import SNet
from models.net_i import INet
from models.net_i101 import INet101
from utils.utils_mine import load_part_of_model2, MaxMinNormalization
import time
from matplotlib import pyplot as plt


torch.manual_seed(2020)

# set which gpu to use
device_id = 0
torch.cuda.set_device(device_id)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'

exp_name = 'VideoSaliency_2021-04-06 23:20:10'

args = {
    'gnn': True,
    'snapshot': '184000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    'input_size': (380, 380),
    'start': 0
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

# to_test = {'ecssd': ecssd_path, 'hkuis': hkuis_path, 'pascal': pascals_path, 'sod': sod_path, 'dutomron': dutomron_path}
# to_test = {'ecssd': ecssd_path}

# to_test = {'davis': os.path.join(davis_path, 'davis_test2')}
# gt_root = os.path.join(davis_path, 'GT')
# flow_root = os.path.join(davis_path, 'flow')
# imgs_path = os.path.join(davis_path, 'davis_test2_single.txt')

# to_test = {'FBMS': os.path.join(fbms_path, 'FBMS_Testset')}
# gt_root = os.path.join(fbms_path, 'GT')
# flow_root = os.path.join(fbms_path, 'FBMS_Testset_flownet2_image')
# imgs_path = os.path.join(fbms_path, 'FBMS_test_single.txt')

# to_test = {'SegTrackV2': os.path.join(segtrack_path, 'SegTrackV2_test')}
# gt_root = os.path.join(segtrack_path, 'GT')
# imgs_path = os.path.join(segtrack_path, 'SegTrackV2_test_single.txt')

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

# to_test = {'MCL': os.path.join(mcl_path, 'MCL_test')}
# gt_root = os.path.join(mcl_path, 'GT')
# imgs_path = os.path.join(mcl_path, 'MCL_test_single.txt')

def main(snapshot):
    # net = R3Net(motion='', se_layer=False, dilation=False, basic_model='resnet50')

    net = INet(cfg=None, GNN=args['gnn'])
    if snapshot is None:
        print ('load snapshot \'%s\' for testing' % args['snapshot'])
        # net.load_state_dict(torch.load('pretrained/R2Net.pth', map_location='cuda:2'))
        # net = load_part_of_model2(net, 'pretrained/R2Net.pth', device_id=2)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                   map_location='cuda:' + str(device_id)))
    else:
        print('load snapshot \'%s\' for testing' % snapshot)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, snapshot + '.pth'),
                                       map_location='cuda:' + str(device_id)))
    net.eval()
    net.cuda()
    results = {}

    with torch.no_grad():

        for name, root in to_test.items():

            precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()

            if args['save_results']:
                check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
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
                    start = time.time()

                    prediction2, prediction, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, prediction3 = net(img_var, flow_var)
                    prediction = torch.sigmoid(prediction)

                    end = time.time()
                    pre_predict = prediction
                    print('running time:', (end - start))
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

                    start = time.time()

                    prediction2, prediction, prediction3, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = net(img_var, flow_var)
                    prediction = torch.sigmoid(prediction3)

                    end = time.time()
                    print('running time:', (end - start))
                    pre_predict = prediction
                # e = Erosion2d(1, 1, 5, soft_max=False).cuda()
                # prediction2 = e(prediction)
                #
                # precision2 = to_pil(prediction2.data.squeeze(0).cpu())
                # precision2 = prediction2.data.squeeze(0).cpu().numpy()
                # precision2 = precision2.resize(shape)
                # prediction2 = np.array(precision2)
                # prediction2 = prediction2.astype('float')

                precision = to_pil(prediction.data.squeeze(0).cpu())
                precision = precision.resize(shape)
                prediction = np.array(precision)
                prediction = prediction.astype('float')

                # plt.style.use('classic')
                # plt.subplot(1, 2, 1)
                # plt.imshow(prediction)
                # plt.subplot(1, 2, 2)
                # plt.imshow(precision2[0])
                # plt.show()

                prediction = MaxMinNormalization(prediction, prediction.max(), prediction.min()) * 255.0
                prediction = prediction.astype('uint8')
                # if args['crf_refine']:
                #     prediction = crf_refine(np.array(img), prediction)

                gt = np.array(Image.open(os.path.join(gt_root, img_name + '.png')).convert('L'))
                precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    precision_record[pidx].update(p)
                    recall_record[pidx].update(r)
                mae_record.update(mae)

                if args['save_results']:
                    folder, sub_name = os.path.split(img_name)
                    save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']), folder)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    Image.fromarray(prediction).save(os.path.join(save_path, sub_name + '.png'))

            fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

            results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

    print ('test results:')
    print (results)
    log_path = os.path.join('result_all.txt')
    if snapshot is None:
        open(log_path, 'a').write(exp_name + ' ' + args['snapshot'] + '\n')
    else:
        open(log_path, 'a').write(exp_name + ' ' + snapshot + '\n')
    open(log_path, 'a').write(str(results) + '\n\n')


if __name__ == '__main__':
    if args['start'] > 0:
        for i in range(args['start'], 204000, 4000):
            main(str(i))
    else:
        main(None)


# BASNet
# {'davis': {'fmeasure': 0.8566843568615541, 'mae': 0.032508174857025014}} 120000
# {'davis': {'fmeasure': 0.8257118647276728, 'mae': 0.08186260435816704}} 100000
# {'davis': {'fmeasure': 0.8162878069424612, 'mae': 0.03628622811730323}} 0
# BASNet2
# {'davis': {'fmeasure': 0.8249547930038379, 'mae': 0.06307644139293314}} 60000
# {'davis': {'fmeasure': 0.8466711413681038, 'mae': 0.03978594446420074}} 100000
# {'davis': {'fmeasure': 0.8372888555262563, 'mae': 0.03733236218858492}} 120000

# gate module insert
# {'davis': {'fmeasure': 0.8564294782400845, 'mae': 0.034741382499633856}} 120000
# {'davis': {'fmeasure': 0.8367139657442704, 'mae': 0.039574989092051545}} 40000

# R3Net (no distillation) VideoSaliency_2020-07-21 13:01:22
# {'davis': {'fmeasure': 0.8562027006431742, 'mae': 0.02722195845054729}} 120000
# {'davis': {'fmeasure': 0.8590793367156063, 'mae': 0.026678991545746208}} 100000
# {'DAVSOD': {'fmeasure': 0.6050681244757543, 'mae': 0.08474575046947992}} 100000
# {'VOS': {'fmeasure': 0.7547719681506946, 'mae': 0.0671947038013288}} 100000

# R3Net (with distillation) VideoSaliency_2020-07-23 01:32:38
# {'davis': {'fmeasure': 0.8708063007796304, 'mae': 0.030973868481739456}} 120000
# {'davis': {'fmeasure': 0.8668223696599545, 'mae': 0.03256905773830208}} 100000
# {'davis': {'fmeasure': 0.7926891841311283, 'mae': 0.05570007050745713}} 0

# {'VOS': {'fmeasure': 0.7804980085734903, 'mae': 0.07218745285743372}} 120000

# gate module insert VideoSaliency_2020-08-24 16:19:00
# {'davis': {'fmeasure': 0.8770755494856973, 'mae': 0.03041000501182174}} 10000
# {'davis': {'fmeasure': 0.878549663716236, 'mae': 0.028963380842353136}} 30000
# {'VOS': {'fmeasure': 0.7901018306975077, 'mae': 0.06622779220610804}} 30000
# {'DAVSOD': {'fmeasure': 0.6739642805392635, 'mae': 0.07081303084017548}} 30000

# CPD
# {'davis': {'fmeasure': 0.8699013545842154, 'mae': 0.03606130829488207}} 120000
# {'davis': {'fmeasure': 0.8719889406088578, 'mae': 0.03487299259319628}} 100000
# {'davis': {'fmeasure': 0.8695780875313027, 'mae': 0.03390185741145803}} 80000
# {'davis': {'fmeasure': 0.800913855008486, 'mae': 0.0404164133863452}} 0

# CPD2
# {'davis': {'fmeasure': 0.8387031680456861, 'mae': 0.03758357432994166}}

# CPD2 only finetune VideoSaliency_2020-08-31 02:01:21
# {'davis': {'fmeasure': 0.8222878832873939, 'mae': 0.041747482896966434}} 60000
# {'davis': {'fmeasure': 0.8385933219114949, 'mae': 0.03462121725596459}} 100000
# {'davis': {'fmeasure': 0.844497796554646, 'mae': 0.03466052830857213}} 80000
# {'davis': {'fmeasure': 0.8437427660140131, 'mae': 0.03307520743178606}} 120000

# {'VOS': {'fmeasure': 0.7569595625226427, 'mae': 0.08371791356920238}} 80000
# {'DAVSOD': {'fmeasure': 0.617840800123522, 'mae': 0.079172990510061}} 80000

# {'VOS': {'fmeasure': 0.7405037052808612, 'mae': 0.07940268278758458}} 60000

# {'DAVSOD': {'fmeasure': 0.6089755250989918, 'mae': 0.0791951545988771}} 100000
# {'VOS': {'fmeasure': 0.7527223544711551, 'mae': 0.08485241885481223}} 100000

# gate module insert VideoSaliency_2020-08-31 11:02:05
# {'davis': {'fmeasure': 0.8546145606079505, 'mae': 0.03725592245174211}}
# {'davis': {'fmeasure': 0.8545848500756481, 'mae': 0.036076494497675145}} 100000
# {'davis': {'fmeasure': 0.8702094216216789, 'mae': 0.03227596775720227}} 8000
# {'davis': {'fmeasure': 0.8721395434349012, 'mae': 0.0326152605046578}} 10000

# {'DAVSOD': {'fmeasure': 0.6246623516711902, 'mae': 0.08521686317336055}} 10000
# {'VOS': {'fmeasure': 0.7891096714831739, 'mae': 0.08420496569470534}} 10000

# DSS
# {'davis': {'fmeasure': 0.7765826462881504, 'mae': 0.04943853521917727}}

# poolnet
# {'davis': {'fmeasure': 0.8641983517948094, 'mae': 0.031427647911369386}} 120000
# {'davis': {'fmeasure': 0.8643076508368502, 'mae': 0.03021199209724235}} 100000
# {'davis': {'fmeasure': 0.8652781719534995, 'mae': 0.030149780373771085}} 80000
# {'davis': {'fmeasure': 0.8629402103793395, 'mae': 0.030353318219999918}} 60000
# {'davis': {'fmeasure': 0.8634947641279994, 'mae': 0.030284757482625123}} 40000
# {'davis': {'fmeasure': 0.8604904754370101, 'mae': 0.030785979459017395}} 20000

# poolnet only finetune
# {'davis': {'fmeasure': 0.8567896379616566, 'mae': 0.028652147056242798}} 10000
# {'davis': {'fmeasure': 0.8530932574012062, 'mae': 0.028328569737440374}} 20000
# {'davis': {'fmeasure': 0.8573124361232648, 'mae': 0.02838192739825468}} 30000

# {'VOS': {'fmeasure': 0.7404455357165366, 'mae': 0.07203330270967374}} 20000
# {'DAVSOD': {'fmeasure': 0.6152740410451761, 'mae': 0.0780330552217813}} 20000

# gate
# {'davis': {'fmeasure': 0.8632113552777682, 'mae': 0.029550991612412567}} 40000
# {'davis': {'fmeasure': 0.8634872265430311, 'mae': 0.029076951244899493}} 30000

# {'VOS': {'fmeasure': 0.7589025935271624, 'mae': 0.07688885680114}} 30000
# {'DAVSOD': {'fmeasure': 0.6359492324304611, 'mae': 0.08098192376104135}}

# RAS

# {'davis': {'fmeasure': 0.8644004769923315, 'mae': 0.03054644717665153}} 80000 + 40000
# {'davis': {'fmeasure': 0.8635395432544891, 'mae': 0.030182278123156766}} 80000 + 20000

# F3Net
# {'davis': {'fmeasure': 0.8653015660248683, 'mae': 0.03404826561811981}} 0
# {'davis': {'fmeasure': 0.8663272724268639, 'mae': 0.029167446537842773}} 120000
# {'davis': {'fmeasure': 0.8704135170765226, 'mae': 0.029225478684095}} 100000
# {'davis': {'fmeasure': 0.8707392987449726, 'mae': 0.028199705166410204}} 80000
# {'davis': {'fmeasure': 0.8749002529630177, 'mae': 0.028620029588870388}} 60000
# {'davis': {'fmeasure': 0.8841000759318469, 'mae': 0.02973330920502805}} 40000
# {'davis': {'fmeasure': 0.8887817824566521, 'mae': 0.030533261610297624}} 20000
# {'davis': {'fmeasure': 0.8890749113651485, 'mae': 0.02940475187227484}}
# {'davis': {'fmeasure': 0.8756644270037889, 'mae': 0.027583954236518913}} no distillation

# {'VOS': {'fmeasure': 0.7758744379092741, 'mae': 0.07942563745765158}} 20000
# {'SegTrackV2': {'fmeasure': 0.8887886543479966, 'mae': 0.025292343551778392}} 20000
# {'DAVSOD': {'fmeasure': 0.6443173444242073, 'mae': 0.08025684179355182}} 20000
# {'DAVSOD': {'fmeasure': 0.6413284895557657, 'mae': 0.07660063219057435}} 60000

# F3Net only finetune VideoSaliency_2020-09-01 07:15:39
# {'davis': {'fmeasure': 0.877378962291034, 'mae': 0.026815478345757576}} 40000
# {'davis': {'fmeasure': 0.8825429675996908, 'mae': 0.026620300446554314}} 30000
# {'davis': {'fmeasure': 0.8785479143849427, 'mae': 0.026080884492927017}} 20000
# {'davis': {'fmeasure': 0.8754949746438053, 'mae': 0.0278916474098535}} 10000

# {'DAVSOD': {'fmeasure': 0.618976746648981, 'mae': 0.07551955168632846}} 10000
# {'VOS': {'fmeasure': 0.7788612473435724, 'mae': 0.06646413065564855}}

# F3Net only finetune VideoSaliency_2020-09-08 08:46:47
# {'VOS': {'fmeasure': 0.7734146998329936, 'mae': 0.07729990310956811}} 2000
# {'davis': {'fmeasure': 0.8656253041652912, 'mae': 0.0348368471552667}} 2000
# {'DAVSOD': {'fmeasure': 0.6294509528059372, 'mae': 0.08523378453630998}} 2000

# sequence training with l2 loss
# {'davis': {'fmeasure': 0.8863832469146518, 'mae': 0.027627839811224686}}
# {'davis': {'fmeasure': 0.8841633873367919, 'mae': 0.02688328257937753}}
# {'davis': {'fmeasure': 0.8858619327344865, 'mae': 0.024967701169178107}}
# {'davis': {'fmeasure': 0.8862040347235166, 'mae': 0.02505401840055184}}

# gate module insert VideoSaliency_2020-08-22 14:17:51
# {'davis': {'fmeasure': 0.889208981681908, 'mae': 0.025872335186497047}}
# {'davis': {'fmeasure': 0.8893343580007665, 'mae': 0.02625041336905838}}
# {'davis': {'fmeasure': 0.892129925183327, 'mae': 0.025956635846254534}} 30000

# {'DAVSOD': {'fmeasure': 0.6241270723640747, 'mae': 0.07644525914911644}} 50000
# {'DAVSOD': {'fmeasure': 0.6269053036187716, 'mae': 0.07612932474782834}} 30000

# {'VOS': {'fmeasure': 0.781705699525925, 'mae': 0.07762718467830418}} 30000
# {'SegTrackV2': {'fmeasure': 0.8769431698390447, 'mae': 0.02364640501186135}} 30000

# R2Net
# {'davis': {'fmeasure': 0.8560587150226311, 'mae': 0.02709158591258486}} 0
# {'davis': {'fmeasure': 0.8740784146143851, 'mae': 0.030000806986006066}} 20000
# {'davis': {'fmeasure': 0.8738029940264168, 'mae': 0.031259597996254655}} 40000
# {'davis': {'fmeasure': 0.8683733244433338, 'mae': 0.032098539788818686}} 60000
# {'davis': {'fmeasure': 0.8708696529215854, 'mae': 0.03109205073446645}} 80000
# {'davis': {'fmeasure': 0.8651841128623684, 'mae': 0.02922830903432984}} 100000

# {'DAVSOD': {'fmeasure': 0.6578043535898054, 'mae': 0.07776031258567824}} 80000
# {'DAVSOD': {'fmeasure': 0.6583439205495094, 'mae': 0.07653715379416723}} 100000
# {'VOS': {'fmeasure': 0.7760511417773258, 'mae': 0.07494954062830472}} 80000
# {'VOS': {'fmeasure': 0.7723011074614098, 'mae': 0.07578435707489597}} 80000

# R2Net only finetune
# {'davis': {'fmeasure': 0.8548179629861553, 'mae': 0.030795675437143337}} 40000
# {'davis': {'fmeasure': 0.8635315735308017, 'mae': 0.030016298981508035}} 30000
# {'davis': {'fmeasure': 0.8625687113609614, 'mae': 0.028724036500521192}} 20000
# {'davis': {'fmeasure': 0.8597927432756874, 'mae': 0.031393451661227506}} 10000

# {'VOS': {'fmeasure': 0.734632638074282, 'mae': 0.0727141879076122}} 10000
# {'VOS': {'fmeasure': 0.7261581810893859, 'mae': 0.07565893406248822}} 30000
# {'DAVSOD': {'fmeasure': 0.6446839646609621, 'mae': 0.07180091274861225}} 30000

# gate module insert
# {'davis': {'fmeasure': 0.8748441653176292, 'mae': 0.027851314948555327}} 10000
# {'davis': {'fmeasure': 0.873028833377723, 'mae': 0.028706954658253268}} 30000

# {'VOS': {'fmeasure': 0.7570477998656433, 'mae': 0.07774102850737559}} 10000
# {'DAVSOD': {'fmeasure': 0.662450524947881, 'mae': 0.07530461510457817}} 10000


# RAS2
# {'davis': {'fmeasure': 0.8229587839207361, 'mae': 0.042331282781968474}} 0 no distillation and no finetune
# {'davis': {'fmeasure': 0.8757726533143092, 'mae': 0.031713322556592864}} 20000
# {'davis': {'fmeasure': 0.877346289325937, 'mae': 0.0311400908295075}} 30000
# {'davis': {'fmeasure': 0.8778071227148782, 'mae': 0.030914894533156396}} 40000
# {'davis': {'fmeasure': 0.8789805277673451, 'mae': 0.03058252114962297}} 50000
# {'davis': {'fmeasure': 0.8614372000730359, 'mae': 0.03390185397691291}} 100000
# {'davis': {'fmeasure': 0.8653076651274525, 'mae': 0.03379165449002227}}

# {'VOS': {'fmeasure': 0.7795804356591828, 'mae': 0.07915214435496876}} 20000

# RAS2 only finetune
# {'davis': {'fmeasure': 0.8605102182763539, 'mae': 0.03238801393782354}}
# {'davis': {'fmeasure': 0.8566328414011626, 'mae': 0.033554104844020126}}

# {'DAVSOD': {'fmeasure': 0.6138489186693313, 'mae': 0.07923908515233896}} 30000
# {'VOS': {'fmeasure': 0.752770620576296, 'mae': 0.07330827505174432}} 30000

# gate module insert
# {'davis': {'fmeasure': 0.8825059020992001, 'mae': 0.030672534257460114}}
# {'davis': {'fmeasure': 0.8829802090654366, 'mae': 0.030159273414776422}}

# {'VOS': {'fmeasure': 0.7774251373208261, 'mae': 0.07819300471733173}} 20000
# {'DAVSOD': {'fmeasure': 0.6414144085751449, 'mae': 0.07907701623820462}} 20000

# F3Net soft loss weight:0.1
# {'davis': {'fmeasure': 0.8826005704242365, 'mae': 0.02458922193742472}} 30000
# {'davis': {'fmeasure': 0.8863944061775805, 'mae': 0.024060505036817413}} 20000
# {'davis': {'fmeasure': 0.8804481307610312, 'mae': 0.025164723752833552}} 10000

# F3Net soft loss weight:0.3
# {'davis': {'fmeasure': 0.8839965558095962, 'mae': 0.02537649761741015}} 30000
# {'davis': {'fmeasure': 0.887562597702381, 'mae': 0.024626356941501025}} 20000
# {'davis': {'fmeasure': 0.8891568096857249, 'mae': 0.02591116499258871}} 10000

# F3Net soft loss weight:0.7
# {'davis': {'fmeasure': 0.8863653285893788, 'mae': 0.027940254455130632}} 30000
# {'davis': {'fmeasure': 0.8893278413504562, 'mae': 0.02743635778153236}} 20000

# F3Net soft loss weight:0.9
# {'davis': {'fmeasure': 0.8920714411828287, 'mae': 0.027868697591622475}} 20000
# {'DAVSOD': {'fmeasure': 0.6618834458323002, 'mae': 0.07729873498680018}}
# {'VOS': {'fmeasure': 0.7989555033147239, 'mae': 0.07746995103929408}}
