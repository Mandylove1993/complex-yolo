import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from . import layer as l
import re
import sys
import numpy as np

class network(nn.Module):
    def __init__(self, layerlist):
        super(network, self).__init__()
        ChannelIn = []
        ChannelOut = []
        self.width, self.height, self.channels, self.lr, self.momentum, self.decay, self.max_batches, self.burn_in, \
            self.policy, self.steps, self.scales ,self.batch= make_input(layerlist[0], ChannelIn, ChannelOut)
        self.widthList = []
        self.heightList = []
        self.widthList.append(self.width)
        self.heightList.append(self.height)
        self.layers = []
        self.layername = []
        self.header = torch.IntTensor([0,0,0])
        self.seen = torch.LongTensor([0])
        for i in range(1, layerlist.__len__()):
            layer = l.make_layer(layerlist[i], self.widthList, self.heightList, ChannelIn, ChannelOut, i-1)
            self.layers.append(layer)
            self.layername.append(layer.name)
        self.layerNum = self.layers.__len__()
        self.models = nn.ModuleList()
        self.module_num = 0
        for i in range(self.layerNum):
            self.models.append(self.layers[i].flow)
            self.module_num += self.layers[i].have_flow

    def forward(self, x):
        input = x
        self.output = []
        for i in range(self.layerNum):
            if self.layername[i] == 'route':
                if self.layers[i].l_in == 0:
                    output = self.output[i + self.layers[i].l_route]
                else:
                    input = torch.cat( (self.output[i + self.layers[i].l_in], self.output[i + self.layers[i].l_route]), 1)
                    output = input
            elif self.layername[i] == 'shortcut':
                output = input + self.output[i + self.layers[i].l_shortcut]
            else:
                output = self.models[i](input)
            self.output.append( output )
            input = output
        self.seen += self.batch
        return output


    def init_weights(self):
        for i in range(self.layerNum):
            if self.layers[i].name == 'conv':
                module = self.layers[i].flow
                if module.__len__() == 2 or module.__len__() == 1:
                    nn.init.orthogonal_(module[0].weight.data)
                elif module.__len__() == 3:
                    nn.init.orthogonal_(module[0].weight.data)
                else:
                    pass
            else:
                pass

    def load_weights(self, weightFile):
        fp = open(weightFile, 'rb')
        if fp is None:
            print('Can not open weight file!')
            sys.exit(0)
        header = np.fromfile(fp, count=3, dtype=np.int32)
        self.header = torch.from_numpy(header)
        seen = np.fromfile(fp, count=1, dtype=np.int64)
        self.seen = torch.from_numpy(seen)
        weightData = np.fromfile(fp, dtype = np.float32)
        fp.close()
        index = 0
        for i in range(self.layerNum):
            if index >= weightData.__len__():
                break
            if self.layers[i].name == 'conv':
                module = self.layers[i].flow
                if module.__len__() == 2 or module.__len__() == 1:
                    index = load_conv_weights(weightData, index, module[0])
                elif module.__len__() == 3:
                    index = load_conv_bn_weights(weightData, index, module[0], module[1])
                else:
                    print(module.__len__())
                    print("Error convolutional type! Load weights failed")
                    sys.exit(-2)
            else:
                pass

    def save_weights(self, weightFile):
        print('saving weight to' + weightFile)
        fp = open(weightFile, 'wb')
        if self.header.is_cuda:
            header = torch.IntTensor(self.header.size()).copy_(self.header)
            seen = torch.LongTensor(self.seen.size()).copy_(self.seen)
        else:
            header = self.header
            seen = self.seen
        header.numpy().tofile(fp)
        seen.numpy().tofile(fp)
        for i in range(self.layerNum):
            if self.layername[i] == 'conv':
                module = self.models[i]
                if module.__len__() <= 2:
                    save_conv_weights(fp, module[0])
                elif module.__len__() == 3:
                    save_conv_bn_weights(fp, module[0], module[1])
                else:
                    print(module.__len__())
                    print("Error convolutional type! Save weights failed")
                    sys.exit(-3)
            else:
                pass
        fp.close()

def load_conv_bn_weights(weightData, index, md_conv, md_bn):
    len_conv = md_conv.weight.numel()
    len_bn = md_bn.bias.numel()
    md_bn.bias.data.copy_( torch.from_numpy( weightData[index:index+len_bn] ) )
    index = index + len_bn
    md_bn.weight.data.copy_( torch.from_numpy(weightData[index:index+len_bn]) )
    index = index + len_bn
    md_bn.running_mean.copy_( torch.from_numpy(weightData[index:index+len_bn]) )
    index = index + len_bn
    md_bn.running_var.copy_( torch.from_numpy(weightData[index:index+len_bn]) )
    index = index + len_bn
    md_conv.weight.data.copy_( torch.from_numpy(weightData[index:index+len_conv]).view(md_conv.weight.data.size()) )
    index = index + len_conv 
    return index

def load_conv_weights(weightData, index, md_conv):
    len_conv = md_conv.weight.numel()
    len_bias = md_conv.bias.numel()
    md_conv.bias.data.copy_( torch.from_numpy( weightData[index:index+len_bias] ) )
    index = index + len_bias
    md_conv.weight.data.copy_( torch.from_numpy(weightData[index:index+len_conv]).view(md_conv.weight.data.size()) )
    index = index + len_conv 
    return index

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def save_conv_bn_weights(fp, md_conv, md_bn):
    if md_bn.bias.is_cuda:
        convert2cpu(md_bn.bias.data).numpy().tofile(fp)
        convert2cpu(md_bn.weight.data).numpy().tofile(fp)
        convert2cpu(md_bn.running_mean).numpy().tofile(fp)
        convert2cpu(md_bn.running_var).numpy().tofile(fp)
        convert2cpu(md_conv.weight.data).numpy().tofile(fp)
    else:
        md_bn.bias.data.numpy().tofile(fp)
        md_bn.weight.data.numpy().tofile(fp)
        md_bn.running_mean.numpy().tofile(fp)
        md_bn.running_var.numpy().tofile(fp)
        md_conv.weight.data.numpy().tofile(fp)

def save_conv_weights(fp, md_conv):
    if md_conv.bias.is_cuda:
        convert2cpu(md_conv.bias.data).numpy().tofile(fp)
        convert2cpu(md_conv.weight.data).numpy().tofile(fp)
    else:
        md_conv.bias.data.numpy().tofile(fp)
        md_conv.weight.data.numpy().tofile(fp)

def make_input(layercfg, ChannelIn, ChannelOut):
    line = layercfg.split('\n')
    p1 = re.compile(r'width=')
    p2 = re.compile(r'height=')
    p3 = re.compile(r'channels=')
    p4 = re.compile(r'learning_rate=')
    p5 = re.compile(r'momentum=')
    p6 = re.compile(r'decay=')
    p7 = re.compile(r'max_batches=')
    p8 = re.compile(r'burn_in=')
    p9 = re.compile(r'policy=')
    p10 = re.compile(r'steps=')
    p11 = re.compile(r'scales=')
    p12 = re.compile(r'batch=')
    width = height = channels = max_batches = burn_in = 0
    lr = momentum = decay = 0.0
    policy = ''
    steps = []
    scales = []
    for info in line:
        if p1.findall(info):
            width = int( re.sub('width=','',info) )
        if p2.findall(info):
            height = int( re.sub('height=','',info) )
        if p3.findall(info):
            channels = int( re.sub('channels=','',info) )
        if p4.findall(info):
            lr = float( re.sub('learning_rate=','',info) )
        if p5.findall(info):
            momentum = float( re.sub('momentum=','',info) )
        if p6.findall(info):
            decay = float( re.sub('decay=','',info) )
        if p7.findall(info):
            max_batches = int( re.sub('max_batches=','',info) )
        if p8.findall(info):
            burn_in = int( re.sub('burn_in=','',info) )
        if p9.findall(info):
            policy = re.sub('policy=','',info)
        if p10.findall(info):
            steps_str = re.sub('steps=','',info).split(',')
            for s in steps_str:
                steps.append( int(s) )
        if p11.findall(info):
            scales_str = re.sub('scales=','',info).split(',')
            for s in scales_str:
                scales.append( float(s) )
        if p12.findall(info):
            batch = int( re.sub('batch=','',info) )
    ChannelIn.append(0)
    ChannelOut.append(channels)
    return width, height, channels, lr, momentum, decay, max_batches, burn_in, policy, steps, scales, batch

    '''
    def forward(self, x):
        input = x
        for i in range(self.layerNum):
            if self.layers[i].name == 'route':
                if self.layers[i].l_in == 0:
                    output = self.layers[i + self.layers[i].l_route].output
                else:
                    input = torch.cat( (self.layers[i + self.layers[i].l_in].output, self.layers[i + self.layers[i].l_route].output), 1)
                    output = input
            elif self.layers[i].name == 'shortcut':
                output = input + self.layers[i + self.layers[i].l_shortcut].output
            else:
                output = self.models[i](input)
            self.layers[i].input = input
            self.layers[i].output = output
            input = output
        return output
        '''