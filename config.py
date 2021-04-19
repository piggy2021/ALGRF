# coding: utf-8
import os

# szu 169 sever
datasets_root = '/home/amax/data/ty'
# local pc
# datasets_root = '/home/qub/data/saliency'

# For each dataset, I put images and masks together
video_train_path = os.path.join(datasets_root, 'Pre-train')
davis_path = os.path.join(datasets_root, 'davis')
visal_path = os.path.join(datasets_root, 'ViSal')
davsod_path = os.path.join(datasets_root, 'DAVSOD')

