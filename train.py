import torch as t
import torch.cuda
import argparse
import os
import sys
import cv2
import re
import model.network as net
import model.loss as loss
import model.data as dat
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
import time
import visdom
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='train a network')
    parser.add_argument('--dataset',help='training set config file',default='dataset/kitti.data',type=str)
    parser.add_argument('--netcfg',help='the network config file',default='cfg/complex-yolo.cfg',type=str)
    parser.add_argument('--weight',help='the network weight file',default='backup/complex-yolo.backup',type=str)
    parser.add_argument('--vis',help='visdom the training process',default=1,type=int)
    parser.add_argument('--init',help='initialize the network parameter',default=0,type=int)
    parser.add_argument('--cuda',help='use the GPU',default=1,type=int)
    parser.add_argument('--ngpus',help='use mult-gpu',default=1,type=int)
    args = parser.parse_args()
    return args

def parse_dataset_cfg(cfgfile):
    with open(cfgfile,'r') as fp:
        p1 = re.compile(r'classes=\d')
        p2 = re.compile(r'train=')
        p3 = re.compile(r'names=')
        p4 = re.compile(r'backup=')
        for line in fp.readlines():
            a = line.replace(' ','').replace('\n','')
            if p1.findall(a):
                classes = re.sub('classes=','',a)
            if p2.findall(a):
                trainlist = re.sub('train=','',a)
            if p3.findall(a):
                namesdir = re.sub('names=','',a)
            if p4.findall(a):
                backupdir = re.sub('backup=','',a)
    return int(classes),trainlist,namesdir,backupdir

def parse_network_cfg(cfgfile):
    with open(cfgfile,'r') as fp:
        layerList = []
        layerInfo = ''
        p = re.compile(r'\[\w+\]')
        p1 = re.compile(r'#.+')
        for line in fp.readlines():
            if p.findall(line):
                if layerInfo:
                    layerList.append(layerInfo)
                    layerInfo = ''
            if line == '\n' or p1.findall(line):
                continue
            line = line.replace(' ','')
            layerInfo += line
        layerList.append(layerInfo)
    print('layer number is %d'%(layerList.__len__() - 1) )
    return layerList

def adjust_learning_rate(optimizer, batch, model, num_gpu):
    lr = model.lr
    for i in range(len(model.steps)):
        scale = model.scales[i] if i < len(model.scales) else 1
        if batch >= model.steps[i]:
            lr = model.lr * scale
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/model.batch*num_gpu
    print('learning rate: %f'%lr)
    return lr

if __name__ == '__main__':
    args = parse_args()
    print(args)
    classes, trainlist, namesdir, backupdir = parse_dataset_cfg(args.dataset)
    print('%d classes in dataset'%classes)
    print('trainlist directory is ' + trainlist)
    #step 1: parse the network
    layerList = parse_network_cfg(args.netcfg)
    netname = args.netcfg.split('.')[0].split('/')[-1]
    layer = []
    print('the depth of the network is %d'%(layerList.__len__()-1))
    network = net.network(layerList)
    max_batch = network.max_batches
    batch = network.batch
    lr = network.lr / batch
    #step 2: load network parameters
    if args.init == 0:
        network.load_weights(args.weight)
    else:
        network.init_weights()
    seen = network.seen
    criterion = loss.CostYoloV2(network.layers[-1].flow[0], seen)
    print('seen=%d'%seen)
    start_batch = seen / batch
    layerNum = network.layerNum
    if args.cuda:
        if args.ngpus:
            print('use mult-gpu')
            network = nn.DataParallel(network).cuda()   
            model = network.module
            num_gpu = torch.cuda.device_count()
        else:
            network = network.cuda()
            model = network
            num_gpu = 1
    #step 3: load data 
    dataset = dat.YoloDataset(trainlist,416,416)
    timesPerEpoch = int(dataset.len / batch)
    max_epoch = int(max_batch / timesPerEpoch)
    print('max epoch : %d'%max_epoch)
    start_epoch = int(start_batch / timesPerEpoch)
    dataloader = data.DataLoader(dataset, batch_size=batch, shuffle=1, drop_last=True)
    #step 4: define optimizer
    optimizer = optim.Adam(network.parameters(),lr=lr*num_gpu)
    #step 5: start train
    print('start training...')
    t_start = time.time()
    #step 6 : initialize visdom board
    if args.vis:
        vis = visdom.Visdom(env=u'test1')
    #step 7 : start training
    for j in range(start_epoch, max_epoch):
        print('start training epoch %d'%j)
        iou_epoch = 0.0
        class_epoch = 0.0
        obj_epoch = 0.0
        for ii,(imgs, labels) in enumerate(dataloader):
            imgs = Variable( imgs )
            labels =  Variable(labels)
            if args.cuda:
                imgs =  imgs.cuda()
                #labels =  labels.cuda()
            #forward propagate
            optimizer.zero_grad()
            t0 = time.time()
            pred = network.forward(imgs)
            #calculate loss
            t1 = time.time()
            pred = pred.cpu()
            cost = criterion(pred, labels)
            cost = cost.cuda()
            #back propagate
            t2 = time.time() 
            cost.backward()
            #update parameters
            t3 = time.time()
            optimizer.step()
            t4 = time.time()
            print('forward time: %f, loss time: %f, backward time: %f, update time: %f'%((t1-t0),(t2-t1),(t3-t2),(t4-t3)))
            loss = criterion.loss.cpu().data.view(1)
            iou_epoch += criterion.Aiou.cpu().data.view(1)
            class_epoch += criterion.AclassP.cpu().data.view(1)
            obj_epoch += criterion.Aobj.cpu().data.view(1)
            if np.isnan(loss.numpy()):
                print('loss is nan, check parameters!')
                sys.exit(-1)
            i = j * timesPerEpoch + ii 
            if i % 500 == 0 and i > 0:
                weightname = backupdir + '/' + netname + '.backup'
                model.save_weights(weightname)
                adjust_learning_rate(optimizer, i, model, num_gpu)
            if args.vis:
                if args.cuda:
                    loss_coords = criterion.loss_coords.cpu().data.view(1)
                    loss_obj = criterion.loss_obj.cpu().data.view(1)
                    loss_noobj = criterion.loss_noobj.cpu().data.view(1)
                    loss_fy = criterion.loss_fy.cpu().data.view(1)
                    loss_classes = criterion.loss_classes.cpu().data.view(1)
                else:
                    loss_coords = criterion.loss_coords.data.view(1)
                    loss_obj = criterion.loss_obj.data.view(1)
                    loss_noobj = criterion.loss_noobj.data.view(1)
                    loss_fy = criterion.loss_fy.data.view(1)
                    loss_classes = criterion.loss_classes.data.view(1)
                if i > 0:
                    vis.line(loss,X=np.array([i]),win='loss',update='append')
                    vis.line(loss_obj,X=np.array([i]),win='loss_obj',update='append')
                    vis.line(loss_noobj,X=np.array([i]),win='loss_noobj',update='append')
                    vis.line(loss_coords,X=np.array([i]),win='loss_coords',update='append')
                    vis.line(loss_fy,X=np.array([i]),win='loss_fy',update='append')
                    vis.line(loss_classes,X=np.array([i]),win='loss_classes',update='append')
                else:
                    vis.line(loss,X=np.array([0]),win='loss',opts=dict(title='loss'))
                    vis.line(loss_obj,X=np.array([0]),win='loss_obj',opts=dict(title='obj_loss'))
                    vis.line(loss_noobj,X=np.array([0]),win='loss_noobj',opts=dict(title='noobj_loss'))
                    vis.line(loss_fy,X=np.array([0]),win='loss_fy',opts=dict(title='fy_loss'))
                    vis.line(loss_coords,X=np.array([0]),win='loss_coords',opts=dict(title='coords_loss'))
                    vis.line(loss_classes,X=np.array([0]),win='loss_classes',opts=dict(title='classes_loss'))
        weightname = backupdir + '/' + netname + '-epoch' + str(j) + '.weight'
        if j%40 == 0:
            model.save_weights(weightname)
        iou_epoch = iou_epoch / timesPerEpoch
        class_epoch = class_epoch / timesPerEpoch
        obj_epoch = obj_epoch / timesPerEpoch
        if args.vis:
            if j > 0:
                vis.line(iou_epoch, X=np.array([j]),win='iou',update='append')
                vis.line(class_epoch, X=np.array([j]),win='class',update='append')
                vis.line(obj_epoch, X=np.array([j]),win='obj',update='append')
            else:
                vis.line(iou_epoch, X=np.array([j]),win='iou',opts=dict(title='iou'))
                vis.line(class_epoch, X=np.array([j]),win='class',opts=dict(title='class'))
                vis.line(obj_epoch, X=np.array([j]),win='obj',opts=dict(title='obj'))
    print('finished training!')
