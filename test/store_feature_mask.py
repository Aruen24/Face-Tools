# -*- coding: utf-8 -*-
import sqlite3
import argparse
import os
import re
import cv2
import numpy as np
import mxnet as mx
import torch
from torchvision import transforms
# from hutian_resnet100 import iresnet100
from hutian_resnet100_mask import iresnet100
from hutian_varGFaceNet_swish_new_mask import varGFaceNet_swish_new
from collections import namedtuple
from sklearn import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_size = [112, 112]

def get_parser():
    parser = argparse.ArgumentParser(description='store freature to database')
    parser.add_argument('--db', type=str, help='sqlite3 database path')
    parser.add_argument('--anchor_pics_path', type=str, help='anchor pic path',
        default='./datasets/imageVIP_anchor')
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    args = parser.parse_args()
    return args

def insert2db(conn, feature, name, pic_name):
    c = conn.cursor()
    feature = " ".join([str(x) for x in feature])
    sql_str = "insert into features(name, feature, pic_name) values('%s', '%s', '%s')" % (name, feature, pic_name)
    c.execute(sql_str)
    conn.commit()

def single_input(path):
    """
    给出图片路径，生成mxnet/pyTorch预测所需固定格式的输入
    :param path: 图片路径
    :return: mxnet/pyTorch所需格式的数据
    """
    img = cv2.imread(path)
    # mxnet三通道输入是严格的RGB格式，而cv2.imread的默认是BGR格式，因此需要做一个转换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # if img.shape[0] != image_size[0]:
        # img = cv2.resize(img, (112, 112))

    # 重塑数组的形态，从（图片高度, 图片宽度, 3）重塑为（3, 图片高度, 图片宽度）
    # img = np.swapaxes(img, 0, 2)
    # img = np.swapaxes(img, 1, 2)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255
    img = img - 0.5
    img = img*2.0

    # 添加一个第四维度并构建NDArray/Tensor
    img = img[np.newaxis, :]
    # mxnet
    # array = mx.nd.array(img)
    # pyTorch
    array = torch.from_numpy(img)
    # print("单张图片输入尺寸：", array.shape)
    return array

TFS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def single_input_mxnet_io(path, transform):
    # 使用mxnet读图片，并转化为PIL格式
    img_string = open(path, 'rb').read()
    img = mx.image.imdecode(img_string, flag=1).asnumpy() # RGB 
    img = transform(img)
    # 添加一个第四维度并构建NDArray/Tensor
    arr = img[np.newaxis, :]
    return arr

    
if __name__ == '__main__':
    args = get_parser()
    
    # mxnet构建并加载GPU模型
    # ctx = mx.gpu(args.gpu)
    
    # #ctx = mx.cpu()

    # prefix = args.model.split(',')[0]
    # epoch = args.model.split(',')[1]

    # print('loading',prefix, epoch)
    # sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
    # all_layers = sym.get_internals()
    # sym = all_layers['fc1_output']
    # #sym = all_layers['resnest_dense0_fwd_output']
    # model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    # model.set_params(arg_params, aux_params)

    # Batch = namedtuple("batch", ['data'])
    
    # pyTorch构建并加载GPU模型
    model = iresnet100(pretrained=False, if_softmax = False, if_l2_norm = False)
    # model = varGFaceNet_swish_new(if_softmax = False, if_l2_norm = False)
    # model = model.cuda() # 这句话紧紧跟在模型建立后一句
    # model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1]) # 测试使用多GPU测试
    # model.load_state_dict(torch.load(args.model))
    pre_dict = torch.load(args.model)
    new_pre = {}
    for k,v in pre_dict.items():
        if k.startswith('module') and not k.startswith('module_list'):
           name = k[7:]
        else:
            name = k
        new_pre[name] = v
    
    model.load_state_dict(new_pre)
    # model = model.cuda().half() # 这句话紧紧跟在模型建立后一句
    model = model.cuda()
    # model.eval() # 只有一次就够了
    
    conn = sqlite3.connect(args.db)
    names = os.listdir(args.anchor_pics_path)
    cnt = 0;
    for name in names:
        name_path = os.path.join(args.anchor_pics_path, name)
        if os.path.isdir(name_path):
            pic_name = os.listdir(name_path)
            #print(pic_name)
            pic_path = os.path.join(name_path, pic_name[0])
            img = single_input(pic_path)
            # img = single_input_mxnet_io(pic_path, TFS)
            # mxnet版本前向提取特征
            # model.forward(Batch([img]), is_train=False)
            # emb = model.get_outputs()[0].asnumpy()
            # emb = preprocessing.normalize(emb)
            # emb = np.squeeze(emb)
            # pyTorch版本前向提取特征
            model.eval() # 只有一次就够了
            with torch.no_grad():
                # img = img.cuda().half()
                img = img.cuda()
                emb = model(img)
                emb = emb.data.cpu().numpy()
                # print(emb)
                emb = preprocessing.normalize(emb)
                emb = np.squeeze(emb)
            insert2db(conn, emb, name, pic_name[0])
            cnt += 1
            if cnt % 1000 == 0:
                print("processed ", cnt)

    conn.close()
