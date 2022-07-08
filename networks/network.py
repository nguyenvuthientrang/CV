import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from .factory import get_backbone, get_aspp, get_head
from arguments import get_argparser
import copy
from collections import OrderedDict

opts = get_argparser().parse_args()

class Net(nn.Module):
    def __init__(self, name, backbone_name, output_stride, pretrained_backbone):
        super(Net, self).__init__()
        self.backbone=get_backbone(name, backbone_name, output_stride, pretrained_backbone)
        self.aspp_layers = nn.ModuleList()
        # self.aspp_layers.append(get_aspp(output_stride))
        self.output_stride = output_stride

        self.heads = nn.ModuleList()
        # self.heads.append(get_head(num_classes))
        self.aux_heads = None

        #add unknown classifiers
        self.heads.append(get_head(1))
        self.aspp_layers.append(get_aspp(output_stride))

        #add background classifiers
        self.heads.append(get_head(1))
        self.aspp_layers.append(get_aspp(output_stride))

        self.num_classes = 2
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)['out']
        # x = [aspp(x) for aspp in self.aspp_layers]
        if False:
            x = torch.cat(x, 1)
            x =  [h(x) for h in self.heads]
            x = torch.cat(x, dim=1)
        else:
            outs = [aspp(x) for aspp in self.aspp_layers]
            x = [head(outs[i]) for i,head in enumerate(self.heads)]
            x = torch.cat(x, dim=1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes, weight_transfer=True):
        self.num_classes += n_classes

        new_aspp = get_aspp(self.output_stride)


        new_head = get_head(n_classes)
        if weight_transfer:
            new_aspp.load_state_dict(self.aspp_layers[0].state_dict())
            new_head[0].load_state_dict(self.heads[0][0].state_dict())
            new_head[1].load_state_dict(self.heads[0][1].state_dict())

            new_head_sd = new_head[3].state_dict()
            old_head_sd = self.heads[0][3].state_dict()
            for i in range(n_classes):
                new_head_sd['weight'][i] = old_head_sd['weight']
                new_head_sd['bias'][i] = old_head_sd['bias']
            new_head[3].load_state_dict(new_head_sd)
            
        self.aspp_layers.append(new_aspp)
        self.heads.append(new_head)

    def freeze(self, freeze=True):
        if freeze:
            training_params = []
            for param in self.parameters():
                param.requires_grad = False

            #last head
            for param in self.aspp_layers[-1].parameters():
                param.requires_grad = True
            for param in self.heads[-1].parameters():
                param.requires_grad = True
            training_params.append({'params': self.heads[-1].parameters(), 'lr': opts.lr})
            training_params.append({'params': self.aspp_layers[-1].parameters(), 'lr': opts.lr})

            #unknown
            for param in self.aspp_layers[0].parameters():
                param.requires_grad = True
            for param in self.heads[0].parameters():
                param.requires_grad = True
            training_params.append({'params': self.heads[0].parameters(), 'lr': opts.lr})
            training_params.append({'params': self.aspp_layers[0].parameters(), 'lr': opts.lr})      
            
            #background
            for param in self.aspp_layers[1].parameters():
                param.requires_grad = True
            for param in self.heads[1].parameters():
                param.requires_grad = True
            training_params.append({'params': self.heads[1].parameters(), 'lr': opts.lr*1e-4})
            training_params.append({'params': self.aspp_layers[1].parameters(), 'lr': opts.lr*1e-4})  
    
        else:
            training_params = [{'params': self.backbone.parameters(), 'lr': 0.001},
                           {'params': self.aspp_layers.parameters(), 'lr': 0.01},
                           {'params': self.heads.parameters(), 'lr': 0.01}]

        return training_params

    # def train(self, mode=True):
    #     super(Net, self).train(mode=mode)
        
    #     if self.bn_freeze:
    #         for m in self.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.eval()
                    
    #                 m.weight.requires_grad = False
    #                 m.bias.requires_grad = False


def _load_model(arch_type, backbone, output_stride, pretrained_backbone):
    if backbone=='mobilenetv2':
        pass
    elif backbone.startswith('resnet'):
        model = Net(arch_type, backbone, output_stride=output_stride, 
                             pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
        
    return model


# Deeplab v3

def deeplabv3_resnet50(output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet(output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone)


# Deeplab v3+

def deeplabv3plus_resnet50(output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone)

    
    
