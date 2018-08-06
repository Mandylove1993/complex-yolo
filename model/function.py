import torch as t
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class reorg(nn.Module):
    def __init__(self, stride):
        super(reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        ChannelOut = channels*(self.stride)*(self.stride)
        out_height = in_height/self.stride
        out_width = in_width/self.stride
        input_view = x.contiguous().view(batch_size, channels, out_height, self.stride, out_width, self.stride)
        shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
        return shuffle_out.view(batch_size, ChannelOut, out_height, out_width)
'''
    def backward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        ChannelOut = channels/(self.stride)/(self.stride)
        out_height = in_height*self.stride
        out_width = in_width*self.stride
        input_view = x.contiguous().view(batch_size, ChannelOut, self.stride, self.stride, in_height, in_width)
        shuffle_out = input_view.permute(0,1,4,2,5,3).contiguous()
        return shuffle_out.view(batch_size, ChannelOut, out_height, out_width)
'''
'''
def reorgFunc(x, stride):
    batch_size, channels, in_height, in_width = x.size()
    ChannelOut = channels*(stride)*(stride)
    out_height = in_height/stride
    out_width = in_width/stride
    input_view = x.contiguous().view(batch_size, channels, out_height, stride, out_width, stride)
    shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
    return shuffle_out.view(batch_size, ChannelOut, out_height, out_width)

class reorg(nn.Module):
    def __init__(self, stride):
        super(reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        return reorgFunc(x, self.stride)
'''
#the output on the channel dim is [x,y,w,h,C,c0,c1,...,cn], the 1th dim of the Tensor(batch x channel x height x width) 
#(x,y,w,h) is the coords, C is the confidence of is_object, c0,c1 ... cn is the classes confidence
class region(nn.Module):
    def __init__(self, classes, coords, num, bias_match, object_scale, noobject_scale, class_scale, coord_scale, anchors):
        super(region, self).__init__()
        self.classes = classes
        self.coords = coords
        self.num = num
        self.out_len = num*(coords + 1 + classes)
        self.bias_match = bias_match
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        self.anchors = anchors
    def forward(self, x):
        outlist = []
        batch_size, channels, in_height, in_width = x.size()
        i = 0
        len_per_anchor = self.coords + 1 + self.classes
        for i in range(self.num):
            index = t.arange(i*len_per_anchor, (i+1)*len_per_anchor, 1)
            index = index.type(t.LongTensor)
            index1 = t.LongTensor([0,1])
            index_wh = t.LongTensor([2,3])
            index2 = t.LongTensor([4])
            index3 = t.arange(5, len_per_anchor, 1)
            index3 = index3.type(t.LongTensor)
            if x.is_cuda:
                index = index.cuda()
                index1 = index1.cuda()
                index_wh = index_wh.cuda()
                index2 = index2.cuda()
                index3 = index3.cuda()
            v = t.index_select(x, 1, index)
            #sigmod the object location(x,y)
            v1 = t.index_select(v, 1, index1)
            out1 = F.sigmoid(v1)
            out_wh = t.index_select(v, 1, index_wh)
            #sigmod the object confidence
            v2 = t.index_select(v, 1, index2)
            out2 = F.sigmoid(v2)
            #softmax the classes confidence
            v3 = t.index_select(v, 1, index3)
            out3 = F.softmax(v3, 1)
            out_buf = t.cat( (out1, out_wh, out2, out3), 1)
            outlist.append(out_buf)
        output = t.cat(outlist, 1)
        return output
        
class yolo(nn.Module):
    def __init__(self, classes, num, mask, anchors, ignore_thresh):
        super(yolo, self).__init__()
        self.classes = classes
        self.num = num
        self.out_len = num*(4 + 1 + classes)
        self.anchors = anchors
        self.ignore_thresh = ignore_thresh
    def forward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        i = 0
        len_per_anchor = 4 + 1 + self.classes
        for i in range(self.num):
            index = t.arange(i*len_per_anchor, (i+1)*len_per_anchor, 1)
            index = index.type(t.LongTensor)
            index1 = t.LongTensor([0,1])
            index_wh = t.LongTensor([2,3])
            index2 = t.LongTensor([4])
            index3 = t.arange(5, len_per_anchor, 1)
            index3 = index3.type(t.LongTensor)
            if x.is_cuda:
                index = index.cuda()
                index1 = index1.cuda()
                index_wh = index_wh.cuda()
                index2 = index2.cuda()
                index3 = index3.cuda()
            v = t.index_select(x, 1, index)
            #sigmod the object location(x,y)
            v1 = t.index_select(v, 1, index1)
            out1 = F.sigmoid(v1)           
            out_wh = t.index_select(v, 1, index_wh)
            #sigmod the object confidence         
            v2 = t.index_select(v, 1, index2)
            out2 = F.sigmoid(v2)
            #yolov3 NO need to softmax the classes confidence
            out3 = t.index_select(v, 1, index3)
            out_buf = t.cat( (out1, out_wh, out2, out3), 1)
            if i == 0:
                output = out_buf
            else:
                output = t.cat((output, out_buf), 1)
        return output