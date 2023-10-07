# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:57:15 2020

@author: Lim
"""
import os
import sys
import torch
import numpy as np
from Loss import CtdetLoss
from Gwd_loss import CtdetGWDLoss                 # 是否加gwd_loss
from torch.utils.data import DataLoader
from dataset import ctDataset
import cfg
sys.path.append(r'./backbone')
from resnet import ResNet
#from resnet_dcn import ResNet
from dlanet import DlaNet
#from dlanet_dcn import DlaNet


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
use_gpu = torch.cuda.is_available()

if cfg.NET == 'ResNet':
    model = ResNet(34)
    model.init_weights(pretrained=True)
else:
    model = DlaNet(34)
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

if cfg.Loss == 'l1':
    loss_weight={'hm_weight':1,'wh_weight':0.1,'ang_weight':0.1,'reg_weight':0.1}
    criterion = CtdetLoss(loss_weight)

else:
    loss_weight = {'hm_weight': 1, 'wh_weight': 0.1, 'ang_weight': 0.1, 'reg_weight': 0.1, 'gwd_weight': 1}
    criterion = CtdetGWDLoss(loss_weight)

print('Using ' + cfg.Loss +'loss...')

device = torch.device("cuda")
if use_gpu:
    model.cuda()
model.train()

learning_rate = cfg.learning_rate
num_epochs = cfg.NUM_EPOCHS
cur_epochs = 0
# 给模块设置不同参数
# params=[]
# params_dict = dict(model.named_parameters())
# for key,value in params_dict.items():
#     params += [{'params':[value],'lr':learning_rate}]

#optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

best_test_loss = np.inf

# 是否断点训练
if cfg.CHECKPOINT == True:
    last_path = './checkpoint/' + cfg.DATASET_NAME + '_' + cfg.Loss +  '/' + 'last.pth'
    cpt = torch.load(last_path)
    learning_rate = cpt['learning_rate']
    cur_epochs = cpt['epoch']
    optimizer.load_state_dict(cpt['optimizer'])  # 优化器导入同理
    model.load_state_dict(cpt['net'])
    best_test_loss = cpt['best_test_loss']
    print('成功断点训练')

train_dataset = ctDataset(split='train')
train_loader = DataLoader(train_dataset,batch_size=2,shuffle=True,num_workers=0)  # num_workers是加载数据（batch）的线程数目

test_dataset = ctDataset(split='val')
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers=0)
print('the dataset has %d images' % (len(train_dataset)))

num_iter = 0

for epoch in range(cur_epochs, num_epochs):
    model.train()
    if epoch == 90:
        learning_rate= learning_rate * 0.1 
    if epoch == 120:
        learning_rate= learning_rate * (0.1 ** 2)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = learning_rate
    if epoch == 250:
        learning_rate = learning_rate * (0.1 ** 3)

    total_loss = 0.
    
    for i, sample in enumerate(train_loader):
        for k in sample:
            sample[k] = sample[k].to(device=device, non_blocking=True)
        pred = model(sample['input'])
        loss = criterion(pred, sample)
        total_loss += loss.item()

        if total_loss >= 1e9:
            break

        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.data, total_loss / (i+1)))
            num_iter += 1

    #validation
    validation_loss = 0.0
    model.eval()
    for i, sample in enumerate(test_loader):
        if use_gpu:
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)
        pred = model(sample['input'])
        loss = criterion(pred, sample)   
        validation_loss += loss.item()
    validation_loss /= len(test_loader)

    # 权重保存路径
    best_path = './checkpoint/' + cfg.DATASET_NAME + '_' + cfg.Loss
    mkdir(best_path)
    last_path = './checkpoint/' + cfg.DATASET_NAME + '_' + cfg.Loss
    mkdir(last_path)

    # 保存权重
    cpt = {
        'net': model.state_dict(),            # 保存模型
        'optimizer': optimizer.state_dict(),  # 保存优化器
        'epoch': epoch,                       # 保存训练轮数
        'learning_rate': learning_rate,
        'best_test_loss': best_test_loss
    }

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        #torch.save(model.state_dict(), best_path +  '/' + 'best.pth')
        torch.save(cpt, best_path +  '/' + 'best.pth')
    #torch.save(model.state_dict(),last_path + '/' + 'last.pth')
    torch.save(cpt, last_path + '/' + 'last.pth')
