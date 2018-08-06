import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Function
import re
import sys
from . import function as F

class Layer(nn.Module):
    def __init__(self, name, order, in_channel, out_channel, layers=None, l_in=-1, l_route=0, l_shortcut=0):
        super(Layer, self).__init__()
        self.name = name
        self.order = order
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.flow = None
        self.have_flow = 0
        if layers:
            self.flow = nn.Sequential(*layers)
            self.have_flow = 1
        self.l_in = l_in
        self.l_route = l_route
        self.l_shortcut = l_shortcut
        self.input = None
        self.output = None
    '''
    def forward(self, x, Layers):       
        if self.l_route != 0:
            if self.l_in == 0:
                self.output = Layers[self.order + self.l_route].output
            else:
                self.input = torch.cat( (Layers[self.order + self.l_in].output, Layers[self.order + self.l_route].output), 1)
                self.output = self.input
        else:
            self.input = x
            if self.l_shortcut != 0:
                self.output = self.input + Layers[self.order + self.l_shortcut].output
            else:
                #self.output = self.flow(self.input)
                pass
        return self.output
    '''

        
def make_conv_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'batch_normalize=\d')
    p2 = re.compile(r'filters=')
    p3 = re.compile(r'size=')
    p4 = re.compile(r'stride=')
    p5 = re.compile(r'pad=')
    p6 = re.compile(r'activation=')
    layers = []
    pad = bn = size = padding_size = 0
    activation = ''
    for info in cfglist:
        if p1.findall(info):
            bn = int( re.sub('batch_normalize=','',info) )
        if p2.findall(info):
            out_channel = int( re.sub('filters=','',info) )
        if p3.findall(info):
            size = int( re.sub('size=','',info))
        if p4.findall(info):
            stride = int( re.sub('stride=','',info) )
        if p5.findall(info):
            pad = int( re.sub('pad=','',info))
        if p6.findall(info):
            activation = re.sub('activation=','',info)
    if pad == 1:
        padding_size = int(size/2)
    in_channel = ChannelOut[order]
    ChannelIn.append(in_channel)
    ChannelOut.append(out_channel)
    if bn:
        layers.append( nn.Conv2d(in_channel, out_channel, size, stride=stride, padding=padding_size, bias=False) )
        layers.append( nn.BatchNorm2d(out_channel) )
    else:
        layers.append( nn.Conv2d(in_channel, out_channel, size, stride=stride, padding=padding_size, bias=True) )
    if activation == 'leaky':
        layers.append( nn.LeakyReLU(negative_slope=0.1, inplace=True) )
    elif activation == 'relu':
        layers.append( nn.ReLU(inplace=True) )
    else:
        pass
    l_conv = Layer('conv', order, in_channel, out_channel, layers=layers)
    w_in = widthlist[order]
    w_out = int((w_in + 2*padding_size - size)/stride + 1)
    h_in = heightlist[order]
    h_out = int((h_in + 2*padding_size - size)/stride + 1)
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'conv2d'
    print('%3d  %8s  %4d  %d x %d / %d  %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,size,size,stride,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_conv

def make_maxpool_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'size=')
    p2 = re.compile(r'stride=')
    p3 = re.compile(r'pad=')
    size = stride = pad = 0
    for info in cfglist:
        if p1.findall(info):
            size = int( re.sub('size=','',info) )
        if p2.findall(info):
            stride = int( re.sub('stride=','',info) )
        if p3.findall(info):
            pad = int( re.sub('pad=','',info) )
    layers = []
    in_channel = ChannelOut[order]
    out_channel = in_channel
    ChannelIn.append(in_channel)
    ChannelOut.append(out_channel)
    w_in = widthlist[order]
    w_out = int((w_in + 2*pad - size)/stride + 1)
    h_in = heightlist[order]
    h_out = int((h_in + 2*pad - size)/stride + 1)
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'maxpool'
    layers.append( nn.MaxPool2d(size, stride=stride, padding=pad) )
    l_pool = Layer('maxpool', order, in_channel, out_channel, layers=layers)
    print('%3d  %8s  %4d  %d / %d      %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,size,stride,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_pool

def make_reorg_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'stride=')
    for info in cfglist:
        if p1.findall(info):
            stride = int( re.sub('stride=','',info) )
    layers = []
    in_channel = ChannelOut[order]
    ChannelIn.append(in_channel)
    out_channel = in_channel*stride*stride
    ChannelOut.append(out_channel)
    w_in = widthlist[order]
    w_out = int(w_in /stride) 
    h_in = heightlist[order]
    h_out = int(h_in /stride)
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'reorg'
    layers.append( F.reorg(stride) )
    l_reorg = Layer('reorg', order, in_channel, out_channel, layers=layers)
    print('%3d  %8s  %4d  %d          %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,stride,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_reorg

def make_route_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'layers=')
    for info in cfglist:
        if p1.findall(info):
            layers_str = re.sub('layers=','',info).split(',')
    if layers_str.__len__() == 1:
        l0 = int(layers_str[0])
        print_str = layers_str[0]
        if l0 > 0:
            l0 = l0 - order
        in_channel = ChannelOut[order + l0 + 1] 
        out_channel = in_channel
        ChannelIn.append(in_channel)
        ChannelOut.append(out_channel)
        l_route = Layer('route', order, in_channel, out_channel, l_in=0, l_route=l0 )
    elif layers_str.__len__() == 2:
        l0 = int(layers_str[0])
        if l0 > 0:
            l0 = l0 - order
        l1 = int(layers_str[1])
        if l1 > 0:
            l1 = l1 - order
        print_str = layers_str[0] + ',' + layers_str[1]
        in_channel = ChannelOut[order + l0 + 1] + ChannelOut[order + l1 + 1]
        out_channel = in_channel
        ChannelIn.append(in_channel)
        ChannelOut.append(out_channel)
        l_route = Layer('route',order, in_channel, out_channel, l_in=l0, l_route=l1 )
    else:
        print('error route layer parameter!')
        sys.exit(1)
    w_in = widthlist[order + l0 + 1]
    w_out = w_in
    h_in = heightlist[order + l0 + 1]
    h_out = h_in
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'route'
    print('%3d  %8s  %4d  %6s     %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,print_str,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_route

def make_upsample_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'stride=')
    stride = 0
    for info in cfglist:
        if p1.findall(info):
            stride = int( re.sub('stride=','',info) )
    layers = []
    in_channel = ChannelOut[order]
    out_channel = in_channel
    ChannelIn.append(in_channel)
    ChannelOut.append(out_channel)
    w_in = widthlist[order]
    w_out = w_in*stride
    h_in = heightlist[order]
    h_out = h_in*stride
    widthlist.append(w_out)
    heightlist.append(h_out)
    name = 'upsample'
    layers.append( nn.UpsamplingNearest2d(scale_factor=stride) )
    l_upsample = Layer('upsample', order, in_channel, out_channel, layers=layers)
    print('%3d  %8s  %4d  %d          %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,stride,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_upsample

def make_shortcut_layers(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'from=')
    p2 = re.compile(r'activation')
    for info in cfglist:
        if p1.findall(info):
            shortcut = int( re.sub('from=','',info) )
        if p2.findall(info):
            activation = re.sub('activation=','',info)
    in_channel = ChannelOut[order] 
    out_channel = in_channel
    ChannelIn.append(in_channel)
    ChannelOut.append(out_channel)
    w_in = widthlist[order]
    w_out = w_in
    h_in = heightlist[order]
    h_out = h_in
    widthlist.append(w_out)
    heightlist.append(h_out)
    if activation == 'linear':
        l_shortcut = Layer('shortcut', order, ChannelIn, ChannelOut, l_shortcut=shortcut)
    else :
        print('unsupport shortcut activation type!')
        sys.exit(0)
    name = 'shortcut' 
    print('%3d  %8s  %4d  %6d     %4d x %4d x %4d  ->  %4d x %4d x %4d'%(order,name,out_channel,shortcut,w_in,h_in,in_channel, w_out,h_out,out_channel))
    return l_shortcut
    

def make_region_layer(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'anchors=')
    p2 = re.compile(r'classes=')
    p3 = re.compile(r'coords=')
    p4 = re.compile(r'num=')
    p5 = re.compile(r'softmax=')
    p6 = re.compile(r'bias_match=')
    p7 = re.compile(r'^object_scale=')
    p8 = re.compile(r'^noobject_scale=')
    p9 = re.compile(r'^class_scale=')
    p10 = re.compile(r'^coord_scale=')
    p11 = re.compile(r'absolute=')
    p12 = re.compile(r'thresh=')
    p13 = re.compile(r'random=')
    p14 = re.compile(r'jitter=')
    p15 = re.compile(r'rescore=')
    anchors = []
    classes = coords = num = softmax = bias_match = object_scale = 0
    noobject_scale = absolute = random = rescore = 0
    thresh = jitter = 0.0
    for info in cfglist:
        if p1.findall(info):
            anchors_str = re.sub('anchors=','',info).split(',')
            for i in range(anchors_str.__len__()):
                anchors.append(float(anchors_str[i]))
        if p2.findall(info):
            classes = int( re.sub('classes=','',info) )
        if p3.findall(info):
            coords = int( re.sub('coords=','',info) )
        if p4.findall(info):
            num = int( re.sub('num=','',info) )
        if p5.findall(info):
            softmax = int( re.sub('softmax=','',info) )
        if p6.findall(info):
            bias_match = int( re.sub('bias_match=','',info) )
        if p7.findall(info):
            object_scale = int( re.sub('object_scale=','',info) )
        if p8.findall(info):
            noobject_scale = int( re.sub('noobject_scale=','',info) )
        if p9.findall(info):
            class_scale = int( re.sub('class_scale=','',info) )
        if p10.findall(info):
            coord_scale = int( re.sub('coord_scale=','',info) )
        if p11.findall(info):
            absolute = int( re.sub('absolute=','',info) )
        if p12.findall(info):
            thresh = float( re.sub('thresh=','',info) )
        if p13.findall(info):
            random = int( re.sub('random=','',info) )
        if p14.findall(info):
            jitter = float( re.sub('jitter=','',info) )
        if p15.findall(info):
            rescore = int( re.sub('rescore=','',info) )
    in_channel = ChannelOut[order]
    ChannelIn.append(in_channel)
    out_channel = in_channel
    ChannelOut.append(out_channel)
    w_in = widthlist[order]
    w_out = w_in 
    h_in = heightlist[order]
    h_out = h_in
    widthlist.append(w_out)
    heightlist.append(h_out)
    layers = []
    layers.append( F.region(classes, coords, num, bias_match, object_scale, noobject_scale, class_scale, coord_scale, anchors ) )
    name = 'detection'
    l_region = Layer('region', order, in_channel, out_channel, layers=layers)
    print('%3d  %10s'%(order,name))
    return l_region

def make_yolo_layer(cfglist, widthlist, heightlist, ChannelIn, ChannelOut, order):
    p1 = re.compile(r'mask=')
    p2 = re.compile(r'anchors=')
    p3 = re.compile(r'classes=')
    p4 = re.compile(r'num=')
    p5 = re.compile(r'jitter=')
    p6 = re.compile(r'ignore_thresh=')
    p7 = re.compile(r'truth_thresh=')
    p8 = re.compile(r'random=')
    anchors = []
    mask = []
    classes = num =  random  = truth_thresh = 0
    ignore_thresh =  jitter = 0.0
    for info in cfglist:
        if p1.findall(info):
            mask_str = re.sub('mask=','',info).split(',')
            for i in range(mask_str.__len__()):
                mask.append(int(mask_str[i]))
        if p2.findall(info):
            anchors_str = re.sub('anchors=','',info).split(',')
            for i in range(anchors_str.__len__()):
                anchors.append(float(anchors_str[i]))
        if p3.findall(info):
            classes = int( re.sub('classes=','',info) )
        if p4.findall(info):
            num = int( re.sub('num=','',info) )
        if p5.findall(info):
            jitter = float( re.sub('jitter=','',info) )
        if p6.findall(info):
            ignore_thresh = float( re.sub('ignore_thresh=','',info) )
        if p7.findall(info):
            truth_thresh = int( re.sub('truth_thresh=','',info) )
        if p8.findall(info):
            random = int( re.sub('random=','',info) )
    in_channel = ChannelOut[order]
    ChannelIn.append(in_channel)
    out_channel = in_channel
    ChannelOut.append(out_channel)
    w_in = widthlist[order]
    w_out = w_in 
    h_in = heightlist[order]
    h_out = h_in
    widthlist.append(w_out)
    heightlist.append(h_out)
    layers = []
    layers.append( F.yolo(classes, num, mask, anchors, ignore_thresh) )
    name = 'yolo'
    l_region = Layer('yolo', order, in_channel, out_channel, layers=layers)
    print('%3d  %8s'%(order,name))
    return l_region

def make_layer(layercfg, widthlist, heightlist, ChannelIn, ChannelOut, order):
    line = layercfg.split('\n')
    layer = None
    if line[0] == '[convolutional]':
        layer = make_conv_layers(line ,widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[maxpool]':
        layer = make_maxpool_layers(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[reorg]':
        layer = make_reorg_layers(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[route]':
        layer = make_route_layers(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[upsample]':
        layer = make_upsample_layers(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[shortcut]':
        layer = make_shortcut_layers(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[region]':
        layer = make_region_layer(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    if line[0] == '[yolo]':
        layer = make_yolo_layer(line, widthlist, heightlist, ChannelIn, ChannelOut, order)
    return layer
