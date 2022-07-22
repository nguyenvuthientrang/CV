"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

from tqdm import tqdm
import networks
import utils
import os
import time
import random
import numpy as np
import cv2

from torch.utils import data
from arguments import get_argparser

import torch
import torch.nn as nn
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
from metrics import StreamSegMetrics

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from networks import network
import itertools
from copy import deepcopy


torch.backends.cudnn.benchmark = True

opts = get_argparser().parse_args()
if opts.approach == 'css':
    from trainers.css import Trainer
elif opts.approach == 'ssul':
    from trainers.ssul import Trainer


log_name = '{}_{}_{}_{}_model_{}_epoch_{}_batchsize_{}_memsize_{}_lr{}'.format(opts.dataset, opts.task, opts.approach, opts.random_seed,
                                                                    opts.model, opts.train_epoch, opts.batch_size, 
                                                                    opts.mem_size, opts.lr)

output = opts.dataset + '/' + opts.task + '/' + opts.approach + '/' + log_name + '.txt'

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

start_step = 0
total_step = len(get_tasks(opts.dataset, opts.task))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("==============================================")
print(f"  task : {opts.task}")
print("  Device: %s" % device)
print( "  opts : ")
print(opts)
print("==============================================")



# Setup random seed
torch.manual_seed(opts.random_seed)
np.random.seed(opts.random_seed)
random.seed(opts.random_seed)

# Set up model
model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
}

model = model_map[opts.model](output_stride=opts.output_stride)
if opts.separable_conv and 'plus' in opts.model:
    network.convert_to_separable_conv(model.classifier)
trainer = Trainer(opts, model, device)

utils.mkdir('checkpoints')

cls_num = len(list(itertools.chain(*list(get_tasks(opts.dataset, opts.task).values()))))
t_num = len(get_tasks(opts.dataset, opts.task))
acc=np.zeros((t_num, cls_num),dtype=np.float32)
miou=np.zeros((t_num, cls_num),dtype=np.float32)
for step in range(start_step, total_step):
    curr_step = step 
    # bn_freeze = opts.bn_freeze if curr_step > 0 else False

    target_cls = get_tasks(opts.dataset, opts.task, curr_step) #total number of class till now except bg and unk
    num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(curr_step+1)] 
    if opts.unknown: # re-labeling: [unknown, background, ...]
        num_classes = [1, 1, num_classes[0]-1] + num_classes[1:] #[1,1,15,1,1,..]
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(curr_step)), #total number of class till be4
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(curr_step+1)) ##total number of class till now
    ]

    print("---------------------------------------------")
    print(f"  step : {curr_step}")
    print("---------------------------------------------")

    # Set up metrics
    metrics = StreamSegMetrics(sum(num_classes)-1 if opts.unknown else sum(num_classes), dataset=opts.dataset)

    # Set up optimizer & parameters
    trainer.add_classes(num_classes[-1])

    print("----------- trainable parameters --------------")
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    print("-----------------------------------------------")

    trainer.train(metrics,curr_idx=curr_idx)
    print("... Training Done")
    class_acc, class_iou = trainer.eval(metrics)
    acc[curr_step,:len(class_acc)] = deepcopy(class_acc)
    miou[curr_step,:len(class_iou)] = deepcopy(class_iou)
print('Save at '+output)
print(acc)
print(miou)
if not os.path.isdir("./result_data/acc/" + opts.dataset + '/' + opts.task + '/' + opts.approach + '/'):
    os.makedirs("./result_data/acc/" + opts.dataset + '/' + opts.task + '/' + opts.approach + '/')
    os.makedirs("./result_data/miou/" + opts.dataset + '/' + opts.task + '/' + opts.approach + '/')
np.savetxt("./result_data/acc/" + output,acc,'%.4f')
np.savetxt("./result_data/miou/" + output,miou,'%.4f')

        