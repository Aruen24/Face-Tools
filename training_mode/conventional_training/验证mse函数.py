# _*_ coding:utf-8 _*_
# 参考：https://blog.csdn.net/hao5335156/article/details/81029791  pytorch的nn.MSELoss损失函数

import torch
import numpy as np

a=np.array([[1,2],[3,4]])
b=np.array([[2,3],[4,1]])


loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)

input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))

loss = loss_fn(input.float(), target.float())

print(loss / 2)
print(loss.type())


'''
loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))

loss = loss_fn(input.float(), target.float())
print(loss)
'''
