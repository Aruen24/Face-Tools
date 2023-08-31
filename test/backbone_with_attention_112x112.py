# _*_ coding:utf-8 _*_

# 划分好训练、验证集，进行训练和验证
import argparse
import os
import time

import numpy as np
# import matplotlib.pyplot as plt
import torch
import torchvision
import torch.backends.cudnn as cudnn

from hutian_varGFaceNet_swish_new import varGFaceNet_swish_new 
from ArcFace import Arcface
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import datasets,transforms,models
from backbone_model_with_attention_112x112 import Net, MobileFaceNet
from PIL import Image
from focal_loss import *
from Speed_up_class import DataLoaderX

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Train on backboneNetwork with Attention Model")
parser.add_argument("--dataDir",default='./',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.1, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--resume', '-r',action='store_true')
args =parser.parse_args()

# use gpu or cpu choice
# device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.dataDir
train_txt = os.path.join(root, 'train.txt')
valid_txt = os.path.join(root, 'val.txt')
train_data_path = "/home/mapengfei/data/product/fruits-360/Training"
val_data_path = "/home/mapengfei/data/product/fruits-360/Test"

#进行图像预处理参数设置
train_transforms = torchvision.transforms.Compose([
    #torchvision.transforms.RandomCrop((112,112),padding=4),
    torchvision.transforms.Resize((112,112)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((112,112)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#自定义数据读写方式
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, flag = '', target_transform = None):
        """
        tex_path : txt文本路径，该文本包含了图像的路径信息，以及标签信息
        transform：数据处理，对图像进行数据增强处理，以及转换成tensor
        """
        self.flag = flag
        fh = open(txt_path,'r') # 读取文件
        imgs = [] # 用来存储路径与标签
        for line in fh:
            line = line.strip('\n')
            if self.flag == 'train' or self.flag == 'val':
                words = line.split(' ')
                imgs.append((words[0], int(words[1])))  # 路径和标签添加到列表中
            elif self.flag == 'test':
                imgs.append(line)
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        if self.flag == 'train' or self.flag == 'val':
            fn, label = self.imgs[index]  # 通过index索引返回一个图像路径fn 与 标签label
            img = Image.open(fn).convert('RGB')  #把图像转成RGB
            if self.transform is not None:
                img = self.transform(img)
            return img, label             #这就返回一个样本
        elif self.flag == 'test':
            fn = self.imgs[index]  # 通过index索引返回一个图像路径fn
            img = Image.open(fn).convert('RGB')  # 把图像转成RGB
            if self.transform is not None:
                img = self.transform(img)
            return img             # 这就返回一个样本

    def __len__(self):
        return len(self.imgs)    # 返回长度，index就会自动的指导读取多少

#trainObj = MyDataset(train_txt, train_transforms, 'train')
#valObj = MyDataset(valid_txt, valid_transforms, 'val')

trainObj = datasets.ImageFolder(train_data_path, transform=train_transforms)
valObj = datasets.ImageFolder(val_data_path, transform=valid_transforms)

# 批次加载图像
# trainloader = torch.utils.data.DataLoader(trainObj, batch_size = 16, shuffle = True)
# validloader = torch.utils.data.DataLoader(valObj, batch_size = 16, shuffle = True)
trainloader = DataLoaderX(trainObj, batch_size = 512, shuffle = True, pin_memory = True, num_workers = 4)
validloader = DataLoaderX(valObj, batch_size = 512, shuffle = True, pin_memory = True, num_workers = 4)

# 构建特征提取网络
num_classes = 120
start_epoch = 0
#net = MobileFaceNet(512, num_classes)
net = varGFaceNet_swish_new(if_softmax = False, if_l2_norm = True, multiplier = 1.25)
head = Arcface(embedding_size=512, classnum=num_classes).cuda()

# 加载预训练模型
if args.resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

net = net.to(device)
net = torch.nn.DataParallel(net.half(), device_ids=[0,1])
head = torch.nn.DataParallel(head.half(), device_ids=[0,1])

# 调参
criterion = FocalLoss(gamma=2).cuda()
# criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD([{'params': net.parameters()}, {'params': head.parameters()}],
                                        lr=0.005, momentum=0.9, weight_decay=1e-4)
best_acc = 0.

# train function for each epoch
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total,
                100. * correct / total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total

# val function for each epoch
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(validloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(validloader), end - start, test_loss / len(validloader), correct, total,
            100. * correct / total
        ))

    # saving checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss / len(validloader), 1. - correct / total
    
# train function for each epoch
def train_feature(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device).half(), labels.to(device)
        pre_features = net(inputs)
        outputs = head.forward(pre_features,labels)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total,
                100. * correct / total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total

# val function for each epoch
def test_feature(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(validloader):
            inputs, labels = inputs.to(device).half(), labels.to(device)
            pre_features = net(inputs)
            outputs = head.forward(pre_features,labels)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(validloader), end - start, test_loss / len(validloader), correct, total,
            100. * correct / total
        ))

    # saving checkpoint
    acc = 100. * correct / total
    # if not os.path.exists('feature_checkpoint'):
            # os.mkdir('features_checkpoint')
    # save
    saved_name = 'Epoch_%d.pt' % epoch
    # state = {'state_dict': model.module.state_dict(), 
             # 'epoch': cur_epoch, 'batch_id': batch_idx}
    state = net.module.state_dict()
    torch.save(state, os.path.join('feature_checkpoint', saved_name))
 
    # if acc > best_acc:
        # best_acc = acc
        # print("Saving parameters to checkpoint/ckpt.t7")
        # checkpoint = {
            # 'net_dict': net.state_dict(),
            # 'acc': acc,
            # 'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
            # os.mkdir('checkpoint')
        # torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss / len(validloader), 1. - correct / total

'''
# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")
'''

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():
    for epoch in range(start_epoch, start_epoch+40):
        train_loss, train_err = train_feature(epoch)
        test_loss, test_err = test_feature(epoch)
        # draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch+1)%15==0:
            lr_decay()

if __name__ == '__main__':
    main()