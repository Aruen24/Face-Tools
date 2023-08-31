"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
import shutil
import argparse
import logging as logger

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset, MXFaceDataset, MXFaceDataset_no_kd, DataLoaderX
from data_processor.Speed_up_class import DataLoaderX
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
#from backbone.hutian_resnet100 import iresnet100
from backbone.hutian_resnet100_mask import iresnet100
from backbone.hutian_varGFaceNet_swish_new_mask import varGFaceNet_swish_new
from head.ArcFace import ArcFace,Arcface

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.backends.cudnn.benchmark = True

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self, backbone_factory, head_factory):
        """Init face model by backbone factorcy and head factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.head = head_factory.get_head()

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        pred = self.head.forward(feat, label)
        # return pred
        return (feat, pred)

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, model, head, optimizer, criterion, cur_epoch, ce_loss_meter, conf):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device).half()
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        if conf.head_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            loss = criterion(outputs, labels) + lamda_lm
        else:
            pre_features = model.forward(images)
            outputs = head.forward(pre_features, labels)
            loss = criterion(outputs, labels)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ce_loss_meter.update(loss.item(), images.shape[0])
        
        if batch_idx % conf.print_freq == 0:
            loss_ce = ce_loss_meter.avg
            
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, loss_ce %f' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_ce))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            
            #conf.writer.add_scalar('Train_loss_ce', loss_ce, global_batch_idx)
            #conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            ce_loss_meter.reset()
        # if (batch_idx + 1) % conf.save_freq == 0:
            # saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            # state = {
                # 'state_dict': model.module.state_dict(),
                # 'epoch': cur_epoch,
                # 'batch_id': batch_idx
            # }
            # torch.save(state, os.path.join(conf.out_dir, saved_name))
            # logger.info('Save checkpoint %s to disk.' % saved_name)
    both_state_name = 'Epoch_model_head_%d.pt' %cur_epoch
    both_state = {'model': model.module.state_dict(), 'head': head.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': cur_epoch}
    torch.save(both_state, os.path.join(conf.out_dir, both_state_name))
    
    saved_name = 'Epoch_%d.pt' % cur_epoch
    # state = {'state_dict': model.module.state_dict(), 
             # 'epoch': cur_epoch, 'batch_id': batch_idx}
    state = model.module.state_dict()
    torch.save(state, os.path.join(conf.out_dir, saved_name))
    logger.info('Save checkpoint %s to disk...' % saved_name)

def train(conf):
    """Total training procedure.
    """
    trainset = MXFaceDataset_no_kd(root_dir=conf.data_root)
    data_loader = DataLoaderX(trainset, batch_size=conf.batch_size, shuffle=True, pin_memory = True, 
                                num_workers = 4, drop_last = True)
    conf.device = torch.device('cuda:0')
    criterion = torch.nn.CrossEntropyLoss().cuda(conf.device)
    # kd_criterion = torch.nn.MSELoss(reduce=True, size_average=False).cuda(conf.device)
    model = iresnet100(pretrained=False, if_softmax = False, if_l2_norm = True).cuda()
    # model = varGFaceNet_swish_new(if_softmax = False, if_l2_norm = True, multiplier = 1.25).cuda()
    #head = ArcFace(feat_dim=512, num_class=416723, margin_arc=0.35, margin_am=0.0, scale=32).cuda()
    head = Arcface(embedding_size=512, classnum=298424).cuda()
    
    ori_epoch = 0
    if conf.resume:
        checkpoint = torch.load(conf.pretrain_model)
        model.load_state_dict(checkpoint['model'])
        #model_dict = model.state_dict()
        #pretrained_dict = checkpoint['model']
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #model_dict.update(pretrained_dict)
        #model.load_state_dict(model_dict)
        print('load model successfully')
        head.load_state_dict(checkpoint['head'])
        print('load head successful')
        ori_epoch = checkpoint['epoch'] + 1
    # model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model.half(), device_ids=[0,1,2,3])
    head = torch.nn.DataParallel(head.half(), device_ids=[0,1,2,3])
    
    # parameters = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(parameters, lr = conf.lr, 
                          # momentum = conf.momentum, weight_decay = 1e-4)
    optimizer = optim.SGD([{'params': model.parameters()}, {'params': head.parameters()}],
                                lr=conf.lr, momentum=conf.momentum, weight_decay=1e-4)
    if conf.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('load optimizer successful')

    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = conf.milestones, gamma = 0.1)

    ce_loss_meter = AverageMeter()
    model.train()
    for epoch in range(ori_epoch, conf.epoches):
        if epoch == 25:
            for params in optimizer.param_groups:
                params['lr'] = 0.001
        elif epoch == 33:
            for params in optimizer.param_groups:
                params['lr'] = 0.0001

        train_one_epoch(data_loader, model, head, optimizer, 
                        criterion, epoch, ce_loss_meter, conf)
        lr_schedule.step()                        

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--data_root", type = str, 
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,  
                      help = "The training file path.")
    conf.add_argument("--backbone_type", type = str, 
                      help = "Mobilefacenets, Resnet, vargfacenet")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type = str, 
                      help = "mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type = str, 
                      help = "the path of head_conf.yaml.")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str, 
                      help = "The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 9, 
                      help = 'The training epoches.')
    conf.add_argument('--step', type = str, default = '2,5,7', 
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 10, 
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type = int, default = 10, 
                      help = 'The save frequency for training state.')
    conf.add_argument('--batch_size', type = int, default = 128, 
                      help='The training batch size over all gpus.')
    conf.add_argument('--momentum', type = float, default = 0.9, 
                      help = 'The momentum for sgd.')
    conf.add_argument('--log_dir', type = str, default = 'log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str, 
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Whether to resume from a checkpoint.')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    #writer = SummaryWriter(log_dir=tensorboardx_logdir)
    #args.writer = writer
    
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
