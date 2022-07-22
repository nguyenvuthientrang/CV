import torch
import os
from copy import deepcopy
from torch import nn 
from torch.utils import data
import time
import utils
from utils.data import get_dataset
from utils.memory import memory_sampling_balanced
from utils.utils import AverageMeter
import numpy as np
from utils.tasks import get_tasks
from utils.mask import *
from arguments import get_argparser
from utils.optimizer import *

opts = get_argparser().parse_args()


class Trainer(object):
    def __init__(self, opts, model, device):
        self.opts=opts
        self.model=model
        self.device=device
        self.model_old = deepcopy(self.model)
        self.optimizer= None
        self.scheduler=None
        self.curr_step=-1
        self.scaler = None
        self.criterion = None

        utils.set_bn_momentum(self.model.backbone, momentum=0.01)
        utils.set_bn_momentum(self.model_old.backbone, momentum=0.01)

        if opts.overlap:
            self.ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
        else:
            self.ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"
        self.fg_idx = 1 if opts.unknown else 0

        self.mask_pre=None
        self.mask_back=None
        self.smax=opts.smax  

        return


    def set_optimizer(self, training_params):
        # if opts.modified_optim:
        if False:
            optimizer = torch.optim.SGD_hat(params=training_params, 
                        lr=self.opts.lr, 
                        momentum=0.9, 
                        weight_decay=self.opts.weight_decay,
                        nesterov=True)
        else:
            optimizer = torch.optim.SGD(params=training_params, 
                                lr=self.opts.lr, 
                                momentum=0.9, 
                                weight_decay=self.opts.weight_decay,
                                nesterov=True)
        return optimizer


    def add_classes(self, n_classes, t=None):
        self.curr_step += 1
        self.model_old = deepcopy(self.model)
        for param in self.model_old.parameters():
            param.requires_grad = False

        #load unknown vao head moi va load aspp gan nhat vao aspp moi
        #freeze model
        if self.curr_step > 0:
            self.model.add_classes(n_classes, True,t)
            training_params = self.model.freeze(True) #freeze and return training params
        else:
            self.model.add_classes(n_classes, False)
            training_params = self.model.freeze(False) #freeze and return training params

        #reset optimizer
        self.optimizer = self.set_optimizer(training_params)
        print("Curr_step:", self.curr_step)
        self.reset_settings()


    def save_ckpt(self, path, best_score):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)   

    def reset_settings(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.opts.amp)
        if self.curr_step > 0:
            if self.opts.mem_size > 0:
                #chinh lai utils
                memory_sampling_balanced(self.opts, self.curr_step, self.model_old)
        self.criterion = self.set_criterion()

        

    def set_data(self):
        if not self.opts.crop_val:
            self.opts.val_batch_size = 1
        
        dataset_dict = get_dataset(self.opts, self.curr_step)
        train_loader = data.DataLoader(
            dataset_dict['train'], batch_size=self.opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = data.DataLoader(
            dataset_dict['val'], batch_size=self.opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = data.DataLoader(
            dataset_dict['test'], batch_size=self.opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

        if self.curr_step > 0 and self.opts.mem_size > 0:
            memory_loader = data.DataLoader(
                dataset_dict['memory'], batch_size=self.opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        else:
            memory_loader = None

        return train_loader, val_loader, test_loader, memory_loader

    def set_criterion(self):
            if self.opts.loss_type == 'focal_loss':
                criterion = utils.FocalLoss(ignore_index=255, size_average=True)
            elif self.opts.loss_type == 'ce_loss':
                criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            elif self.opts.loss_type == 'bce_loss':
                criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, 
                                                                reduction='mean')
                
            return criterion

    def set_scheduler(self, total_itrs):
        if self.opts.lr_policy=='poly':
            scheduler = utils.PolyLR(self.optimizer, total_itrs, power=0.9)
        elif self.opts.lr_policy=='step':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts.step_size, gamma=0.1)
        elif self.opts.lr_policy=='warm_poly':
            warmup_iters = int(total_itrs*0.1)
            scheduler = utils.WarmupPolyLR(self.optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)
        else:
            return
        return scheduler


    def train(self, metrics, curr_idx, thres_cosh=50,thres_emb=6):
        self.model = self.model.to(self.device)
        self.model.train()
        self.parallel_model = nn.DataParallel(self.model)
        self.parallel_model = self.parallel_model.to(self.device)
        self.parallel_model.train()


        if self.model_old != None:
            self.model_old = self.model_old.to(self.device)
            self.model_old.eval()
            self.parallel_model_old = nn.DataParallel(self.model_old)
            self.parallel_model_old = self.parallel_model_old.to(self.device)
            self.parallel_model_old.eval()

        self.train_loader, self.val_loader, self.test_loader, self.memory_loader = self.set_data()

        total_itrs = self.opts.train_epoch * len(self.train_loader)
        val_interval = 40 #max(100, total_itrs // 100)
        print(f"... train epoch : {self.opts.train_epoch} , iterations : {total_itrs} , val_interval : {val_interval}")

        avg_loss = AverageMeter()
        avg_time = AverageMeter()
        self.scheduler = self.set_scheduler(total_itrs)

        best_score = -1
        cur_itrs = 0
        cur_epochs = 0

        #loop
        while cur_itrs < total_itrs:
            cur_itrs += 1
            self.optimizer.zero_grad()
            end_time = time.time()
            
            """ data load """
            try:
                images, labels, sal_maps, _ = train_iter.next()
            except:
                train_iter = iter(self.train_loader)
                images, labels, sal_maps, _ = train_iter.next()
                cur_epochs += 1
                avg_loss.reset()
                avg_time.reset()
                
            images = images.to(self.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
            sal_maps = sal_maps.to(self.device, dtype=torch.long, non_blocking=True)
            s=(self.smax-1/self.smax)*((cur_itrs%len(self.train_loader)+1)/len(self.train_loader))+1/self.smax
            # print("\nsmax:", self.smax)
            # print("Batch:", cur_itrs%len(self.train_loader))
            # print("Total batch:", len(self.train_loader))
            # print("s:", s)


            if self.curr_step > 0 and self.opts.mem_size > 0:
                try:
                    m_images, m_labels, m_sal_maps, _ = mem_iter.next()
                except:
                    mem_iter = iter(self.memory_loader)
                    m_images, m_labels, m_sal_maps, _ = mem_iter.next()

                m_images = m_images.to(self.device, dtype=torch.float32, non_blocking=True)
                m_labels = m_labels.to(self.device, dtype=torch.long, non_blocking=True)
                m_sal_maps = m_sal_maps.to(self.device, dtype=torch.long, non_blocking=True)
                
                rand_index = torch.randperm(self.opts.batch_size)[:self.opts.batch_size // 2].cuda()
                images[rand_index, ...] = m_images[rand_index, ...]
                labels[rand_index, ...] = m_labels[rand_index, ...]
                sal_maps[rand_index, ...] = m_sal_maps[rand_index, ...]

            """ forwarding and optimization """
            with torch.cuda.amp.autocast(enabled=self.opts.amp):
                
                outputs, mask = self.parallel_model(images, self.curr_step, s=s)

                if self.opts.pseudo and self.curr_step > 0:
                    """ pseudo labeling """
                    # print("Pseudo label")
                    with torch.no_grad():
                        outputs_prev, _ = self.parallel_model_old(images, self.curr_step-1, s=self.smax)

                    if self.opts.loss_type == 'bce_loss':
                        pred_prob = torch.sigmoid(outputs_prev).detach()
                    else:
                        pred_prob = torch.softmax(outputs_prev, 1).detach()
                        
                    pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                    pseudo_labels = torch.where( (labels <= self.fg_idx) & (pred_labels > self.fg_idx) & (pred_scores >= self.opts.pseudo_thresh), 
                                                pred_labels, 
                                                labels)
                    
                    # print("Pseudo labels:", pseudo_labels)
                    # print("Pseudo labels:", outputs)
                    loss = self.criterion(outputs, pseudo_labels) + hat_reg(self.mask_pre,mask)
                else:
                    loss = self.criterion(outputs, labels) + hat_reg(self.mask_pre,mask)
                

                self.scaler.scale(loss).backward()

                #Restrict layer gradients in backprop
                if self.curr_step>0:
                    for n,p in self.model.named_parameters():
                        if n in self.mask_back:
                            p.grad.data*=self.mask_back[n]
                            print(p.grad.data)

                #Compensate embedding gradients
                for n, p in self.model.named_parameters():
                    if 'ec' in n:
                        if p.grad is not None:
                            num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                            den = torch.cosh(p.data) + 1
                            p.grad *= self.smax / s * num / den                


                self.scaler.step(self.optimizer)
                self.scaler.update()

                #Constrain embeddings
                for n, p in self.model.named_parameters():
                    if 'ec' in n:
                        if p.grad is not None:
                            p.data.copy_(torch.clamp(p.data, -thres_emb, thres_emb))

                self.scheduler.step()
                avg_loss.update(loss.item())
                avg_time.update(time.time() - end_time)
                end_time = time.time()

                if (cur_itrs) % 10 == 0:
                    print(s)
                    print("[%s / step %d] Epoch %d, Itrs %d/%d, Loss=%6f, Time=%.2f , LR=%.8f" %
                        (self.opts.task, self.curr_step, cur_epochs, cur_itrs, total_itrs, 
                        avg_loss.avg, avg_time.avg*1000, self.optimizer.param_groups[0]['lr']))

                if val_interval > 0 and (cur_itrs) % val_interval == 0:
                    print("validation...")
                    self.model.eval()
                    self.parallel_model.eval()
                    val_score = self.validate(metrics, self.val_loader)
                    print(metrics.to_str(val_score))
                    
                    self.model.train()
                    self.parallel_model.train()
                    
                    class_iou = list(val_score['Class IoU'].values())
                    val_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] + [class_iou[0]])
                    curr_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] )
                    print("curr_val_score : %.4f" % (curr_score))
                    print()
                    
                    if curr_score > best_score:  # save best model
                        print("... save best ckpt : ", curr_score)
                        best_score = curr_score
                        self.save_ckpt(self.ckpt_str % (self.opts.model, self.opts.dataset, self.opts.task, self.curr_step), best_score)
    
        #update cumulative mask and get the backward mask
        self.mask_pre = cum_mask(self.model, self.mask_pre, self.curr_step, self.smax)
        self.mask_back = freeze_mask(self.model, self.mask_pre, self.curr_step, self.smax)

    def eval(self, metrics):
        # if self.curr_step > 0:
        print("... Testing Best Model")
        best_ckpt = self.ckpt_str % (self.opts.model, self.opts.dataset, self.opts.task, self.curr_step)
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        self.parallel_model = nn.DataParallel(self.model)
        self.parallel_model.eval()
        
        test_score = self.validate(metrics=metrics, loader=self.test_loader)
        print(metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(self.opts.dataset, self.opts.task, 0))

        if self.curr_step > 0:

            print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
            print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
            print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
            print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))    
            print(f"...overall mIOU: %.6f" % np.mean(class_iou))   
            print(f"...overall acc: %.6f" % np.mean(class_acc))  
        else:
            print(f"...overall mIOU: %.6f" % np.mean(class_iou))   
            print(f"...overall acc: %.6f" % np.mean(class_acc))      
        return class_acc, class_iou

    def validate(self, metrics, loader):
        """Do validation and return specified samples"""
        metrics.reset()
        ret_samples = []

        with torch.no_grad():
            for i, (images, labels, _, _) in enumerate(loader):
                
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                
                outputs, _ = self.parallel_model(images, self.curr_step, s=self.smax)
                
                if self.opts.loss_type == 'bce_loss':
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=1)
                        
                # remove unknown label
                if self.opts.unknown:
                    outputs[:, 1] += outputs[:, 0]
                    outputs = outputs[:, 1:]
                
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                metrics.update(targets, preds)
                    
            score = metrics.get_results()
        return score