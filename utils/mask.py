import torch
from arguments import get_argparser
import numpy as np

opts = get_argparser().parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def cum_mask(model, p_mask, t, smax):
    """ 
        Keep track of mask values. 
        This will be used later as a regularizer in the optimization
    """
    try:
        model = model.module
    except AttributeError:
        model = model

    task_id = torch.tensor([t]).to(device)
    mask = {}
    for n, _ in model.backbone.named_parameters():
        names = n.split('.')
        checker = [i for i in ['ec0', 'ec1', 'ec2', 'ec3'] if i in names]
        if names[0] == 'module':
            names = names[1:]
        if checker:
            if 'layer' in n:
                gc1, gc2, gc3 = model.backbone.__getattr__(names[0])[int(names[1])].mask(task_id, s=smax)
                if checker[0] == 'ec1':
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = gc1.detach()
                    mask[n].requires_grad = False
                elif checker[0] == 'ec2':
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = gc2.detach()
                    mask[n].requires_grad = False  
                elif checker[0] == 'ec3':
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = gc3.detach()
                    mask[n].requires_grad = False            
            elif checker[0] == 'ec0':
                n = '.'.join(n.split('.')[:-1])
                mask[n] = model.backbone.mask(task_id, smax).detach()
                mask[n].requires_grad = False
    if p_mask is None:
        p_mask = {}
        for n in mask.keys():
            p_mask[n] = mask[n]
    else:
        for n in mask.keys():
            p_mask[n] = torch.max(p_mask[n], mask[n])
    return p_mask

def freeze_mask(model, p_mask, t, smax):
    """
        Eq (2) in the paper. self.mask_back is a dictionary whose keys are
        the convolutions' parameter names. Each value of a key is a matrix, whose elements are
        approximately binary.
    """
    try:
        model = model.module
    except AttributeError:
        model = model

    mask_back = {}
    len_layers = [3, 4, 23, 3]
    for n, p in model.backbone.named_parameters():
        names = n.split('.')
        if 'layer' not in names[0]:
            if n == 'conv1.weight':
                # print(p.shape)
                # print(p_mask['ec0'].shape)
                mask_back[n] = 1 - p_mask['ec0'].data.view(-1, 1, 1, 1).expand_as(p)
        if 'conv' in n and 'layer' in n:
            if 'layer1' in names[0]:
                if names[1] == '0':
                    if names[2]=='conv1':
                        post = '.'.join(names[0:2]) + '.ec1'
                        pre = 'ec0'
                    else:
                        post = '.'.join(names[0:2]) + '.ec' + str(int(names[2][-1]))
                        pre = '.'.join(names[0:2]) + '.ec' + str(int(names[2][-1])-1)
                    
                else:
                    if names[2]=='conv1':
                        post = '.'.join(names[0:2]) + '.ec1'
                        pre = names[0] + '.' + str(int(names[1])-1) + '.ec3'

                    else:
                        post = '.'.join(names[0:2]) + '.ec' + str(int(names[2][-1]))
                        pre = '.'.join(names[0:2]) + '.ec' + str(int(names[2][-1])-1)

            else:
                if names[1] == '0':
                    if names[2]=='conv1':
                        post = '.'.join(names[0:2]) + '.ec1'
                        pre = 'layer' + str(int(names[0][-1])-1) + '.' + str(len_layers[int(names[0][-1])-2]-1) + '.ec3'

                    else:
                        post = '.'.join(names[0:2]) + '.ec' + str(int(names[2][-1]))
                        pre = '.'.join(names[0:2]) + '.ec' + str(int(names[2][-1])-1)
                    
                else:
                    if names[2]=='conv1':
                        post = '.'.join(names[0:2]) + '.ec1'
                        pre = names[0] + '.' + str(int(names[1])-1) + '.ec3'
                    else:
                        post = '.'.join(names[0:2]) + '.ec' + str(int(names[2][-1]))
                        pre = '.'.join(names[0:2]) + '.ec' + str(int(names[2][-1])-1)
            post = p_mask[post].data.view(-1, 1, 1, 1).expand_as(p)
            pre  = p_mask[pre].data.view(1, -1, 1, 1).expand_as(p)
            mask_back[n] = 1 - torch.min(post, pre)
    return mask_back

def hat_reg(p_mask, masks):
    """ masks and p_mask must have values in the same order """
    reg, count = 0., 0.
    if p_mask is not None:
        for m, mp in zip(masks, p_mask.values()):
            aux = 1. - mp#.to(device)
            reg += (m * aux).sum()
            count += aux.sum()
        reg /= count
        return opts.lamb * reg
    else:
        for m in masks:
            reg += m.sum()
            count += np.prod(m.size()).item()
        reg /= count
        return opts.lamb * reg