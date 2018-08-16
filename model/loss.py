import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Function
from torch.autograd import Variable
from . import function as F
import time
import copy

def box_iou(box1, box2):
#box is a [x,y,w,h] tensor
    w = Variable(torch.Tensor([[1,0,-0.5,0],[0,1,0,-0.5],[1,0,0.5,0],[0,1,0,0.5]]))
    if box1.is_cuda:
        w = w.cuda()
    box_a = w.mm(box1.view(4,1))
    box_b = w.mm(box2.view(4,1))
    left = torch.max(box_a[0], box_b[0])
    right = torch.min(box_a[2], box_b[2])
    up = torch.max(box_a[1], box_b[1])
    down = torch.min(box_a[3], box_b[3])
    intersection_w = right - left
    intersection_h = down - up
    if intersection_w < 0 or intersection_h < 0:
        return 0
    else:
        intersection = intersection_w * intersection_h
        union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
        return intersection/union

#this is the vectorized box IOU calculation method
def box_iou_truth(pred_box, truth_box):
    num_pred, num_coords = pred_box.size()
    assert(num_coords == 4)
    num_truth, num_coords = truth_box.size()
    assert(num_coords == 4)
    w = Variable(torch.Tensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]))
    zero = Variable(torch.zeros(num_truth))
    iou = Variable(torch.zeros(num_pred))
    if pred_box.is_cuda:
        w = w.cuda()
        zero = zero.cuda()
        iou = iou.cuda()
    truth_box_ex = truth_box.mm(w)
    pred_box_ex = pred_box.mm(w)
    pred_box_ex2 = pred_box_ex.view(1,4*num_pred).expand(num_truth,4*num_pred)
    pred_box_ex3 = pred_box_ex2.view(num_truth,num_pred,4).permute(1,0,2)
    truth_box_ex3 = truth_box_ex.expand(num_pred, num_truth, 4)
    left = torch.max(pred_box_ex3[:,:,0], truth_box_ex3[:,:,0])
    right = torch.min(pred_box_ex3[:,:,2], truth_box_ex3[:,:,2])
    up = torch.max(pred_box_ex3[:,:,1], truth_box_ex3[:,:,1])
    down = torch.min(pred_box_ex3[:,:,3], truth_box_ex3[:,:,3])
    intersection_w = torch.max( right.sub(left), zero)
    intersection_h = torch.max( down.sub(up), zero)
    intersection = intersection_w.mul(intersection_h)
    w_truth = truth_box[:,2].view(1,num_truth).expand(num_pred, num_truth)
    h_truth = truth_box[:,3].view(1,num_truth).expand(num_pred, num_truth)
    w_pred = pred_box[:,2].view(num_pred,1).expand(num_pred, num_truth)
    h_pred = pred_box[:,3].view(num_pred,1).expand(num_pred, num_truth)
    union = torch.add(w_truth.mul(h_truth), w_pred.mul(h_pred)).sub(intersection)
    iou,idx = intersection.div(union).max(dim=1)
    return iou, idx

#this is the iou vectorized calculation for the loss function, pred_box must has the same length of truth box
def box_iou_v(pred_box, truth_box):
    num, num_pred, num_coords = pred_box.size()
    assert(num_coords == 4)
    num_truth, num_coords = truth_box.size()
    assert(num_coords == 4)
    assert(num_pred == num_truth)
    w = Variable(torch.Tensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]))
    zero = Variable(torch.zeros(num_truth))
    iou = Variable(torch.zeros(num_pred))
    if pred_box.is_cuda:
        w = w.cuda()
        zero = zero.cuda()
        iou = iou.cuda()
    pred_box_ex = torch.mm(pred_box.view(num*num_pred, 4), w).view(num, num_pred, num_coords)
    truth_box_ex = truth_box.mm(w).view(1,num_truth, 4).expand(num, num_truth, 4)
    left = torch.max(pred_box_ex[:,:,0], truth_box_ex[:,:,0])
    right = torch.min(pred_box_ex[:,:,2], truth_box_ex[:,:,2])
    up = torch.max(pred_box_ex[:,:,1], truth_box_ex[:,:,1])
    down = torch.min(pred_box_ex[:,:,3], truth_box_ex[:,:,3])
    intersection_w = torch.max( right.sub(left), zero)
    intersection_h = torch.max( down.sub(up), zero)
    intersection = intersection_w.mul(intersection_h)
    w_truth = truth_box[:,2].view(1,num_truth).expand(num, num_truth)
    h_truth = truth_box[:,3].view(1,num_truth).expand(num, num_truth)
    w_pred = pred_box[:,:,2]
    h_pred = pred_box[:,:,3]
    union = torch.add(w_truth.mul(h_truth), w_pred.mul(h_pred)).sub(intersection)
    iou,idx = intersection.div(union).max(dim=0)
    return iou, idx


class CostYoloV2(nn.Module):
    def __init__(self, RegionLayer, seen):
        super(CostYoloV2, self).__init__()
        self.count = 0
        self.iou = 0.0
        self.recall = 0.0
        self.precision = 0.0
        self.obj = 0.0
        self.noobj = 0.0
        self.obj_pred_count = 0
        self.anchors = RegionLayer.anchors
        self.object_scale = RegionLayer.object_scale
        self.noobject_scale = RegionLayer.noobject_scale
        self.class_scale = RegionLayer.class_scale
        self.coord_scale = RegionLayer.coord_scale
        self.classes = RegionLayer.classes
        self.coords = RegionLayer.coords
        self.num = RegionLayer.num
        self.anchor_len = self.classes + self.coords + 1
        self.thresh = 0.5
        self.seen = seen
    def forward(self, x, truth):
        batch_size, channels, in_height, in_width = x.size()
        self.seen += 1
        self.count = 0
        self.obj_pred_count = 0
        self.iou = 0.0
        self.recall = 0.0
        self.precision = 0.0
        self.class_precision = 0.0
        self.obj = 0.0
        self.noobj = 0.0
        self.Aiou = 0.0
        self.AclassP = 0.0
        self.Aobj = 0.0
        self.Anoobj = 0.0
        self.Arecall = 0.0
        self.loss_noobj = Variable(torch.zeros(1),requires_grad=True)
        self.loss_obj = Variable(torch.zeros(1),requires_grad=True)
        self.loss_coords = Variable(torch.zeros(1),requires_grad=True)
        self.loss_fy = Variable(torch.zeros(1),requires_grad=True)
        self.loss_classes = Variable(torch.zeros(1),requires_grad=True)
        self.loss = Variable(torch.zeros(1),requires_grad=True)
        truth_tensor = Variable(torch.zeros(x.size()))
        zeros = Variable(torch.zeros(1))
        one = Variable(torch.ones(1))
        width_v = torch.arange(0,in_width).view(1,in_width)
        height_v = torch.arange(0,in_height).view(in_height,1)
        if x.is_cuda:
            truth_tensor = truth_tensor.cuda()
            width_v = width_v.cuda()
            height_v = height_v.cuda()
            zeros = zeros.cuda()
            one = one.cuda()
            self.loss_noobj = self.loss_noobj.cuda()
            self.loss_obj = self.loss_obj.cuda()
            self.loss_coords = self.loss_coords.cuda()
            self.loss_fy = self.loss_fy.cuda()
            self.loss_classes = self.loss_classes.cuda()
            self.loss = self.loss.cuda()
        mse = nn.MSELoss(size_average=False)
        truth_noobj_list = []
        noobj_pred_list = []
        coords_pred_list = []
        box_pred_list = []
        box_truth_list = []
        euler_pred_list= []
        euler_truth_list = []
        obj_pred_list = []
        classes_pred_list = []
        classes_truth_list = []
        i_list = width_v.expand(in_height,in_width).contiguous().view(in_height*in_width)
        j_list = height_v.expand(in_height,in_width).contiguous().view(in_height*in_width)
        for b in range(batch_size):
            x_b = x[b, :, :, :]
            truth_b = truth[b, :, :]           
            #get the no object loss
            for n in range(self.num):
                idx = n * self.anchor_len
                idx_end = idx + 4
                box_pred = x_b[idx:idx_end, :, :]
                box_pred = box_pred.clone()
                obj_pred = x_b[idx+6, :, :]
                obj_pred = obj_pred.clone()
                box_truth = truth_b[:, 0:4]
                box_truth = box_truth.clone()
                box_pred_v = box_pred.permute(1,2,0).contiguous().view(in_height*in_width, 4)
                coords_pred_list.append(box_pred_v.view(1, in_height*in_width, 4))
                box_pred_v[:,0] = box_pred_v[:,0].add(i_list).div(in_width)
                box_pred_v[:,1] = box_pred_v[:,1].add(j_list).div(in_height)
                box_pred_v[:,2] = torch.exp(box_pred_v[:,2]).mul(self.anchors[n*2]/in_width)
                box_pred_v[:,3] = torch.exp(box_pred_v[:,3]).mul(self.anchors[n*2+1]/in_height)
                iou, idx_t = box_iou_truth(box_pred_v, box_truth)
                iou = iou.view(in_height, in_width)
                truth_tensor_noobj = iou.sub(self.thresh).sign()
                truth_tensor_noobj = torch.max(truth_tensor_noobj, zeros)
                self.noobj +=  truth_tensor_noobj.sum()
                truth_tensor = obj_pred.mul(truth_tensor_noobj).view(1, in_height, in_width)
                noobj_pred_list.append(obj_pred.view(1, in_height, in_width))
                truth_noobj_list.append(truth_tensor)
            #get object , coords and classes loss    
            for t in range(50):
                box_truth = truth_b[t, 0:4].view(4)
                class_truth = int( truth_b[t, 7].view(1) )
                if box_truth[2] == 0.0 or box_truth[3] == 0.0:
                    break
                best_iou = 0.0
                best_n = 0
                i = int(box_truth[0] * in_width)
                j = int(box_truth[1] * in_height)
                #find the best iou for the current label
                box_truth_shift = box_truth
                box_truth_shift = box_truth_shift.clone()
                box_truth_shift[0] = 0.0
                box_truth_shift[1] = 0.0
                box_pred_shift_l = []
                for n in range(self.num):
                    box_pred = x_b[(n*self.anchor_len):(n*self.anchor_len+4), j, i].view(4)
                    box_pred = box_pred.clone()
                    box_pred_shift = box_pred
                    box_pred_shift[0] = 0.0
                    box_pred_shift[1] = 0.0
                    box_pred_shift[2] = torch.exp(box_pred_shift[2]) * (self.anchors[n*2]/in_width)
                    box_pred_shift[3] = torch.exp(box_pred_shift[3]) * (self.anchors[n*2 + 1]/in_height)
                    box_pred_shift_l.append(box_pred_shift.view(1,4))
                box_pred_shift = torch.cat(box_pred_shift_l, 0)
                iou, best_n = box_iou_truth(box_truth_shift.view(1,4), box_pred_shift )
                best_n = int(best_n)
                #calculate the coords loss
                tx = box_truth[0] * in_width - i
                ty = box_truth[1] * in_height -j
                tw = torch.log( box_truth[2] * in_width / self.anchors[2*best_n] )
                th = torch.log( box_truth[3] * in_height / self.anchors[2*best_n + 1])
                box_t = Variable(torch.Tensor([tx,ty,tw,th]).view(1,4))
                if x.is_cuda:
                    box_t = box_t.cuda()
                box_best = x_b[(best_n*self.anchor_len):(best_n*self.anchor_len+4), j, i].view(1,4)
                box_pred_list.append(box_best)
                box_truth_list.append(box_t)
                if iou > 0.5:
                    self.recall += 1
                self.iou += iou
                #calculate the Euler loss
                tim = truth_b[t, 5]
                tre = truth_b[t, 6]
                box_fy = Variable(torch.Tensor([tim,tre]).view(1,2))
                if x.is_cuda:
                    box_fy = box_fy.cuda()
                euler_best = x_b[(best_n*self.anchor_len+4):(best_n*self.anchor_len+6), j, i].view(1,2)
                euler_pred_list.append(euler_best)
                euler_truth_list.append(box_fy)
                #calculate the object loss
                obj_pred = x_b[(6 + best_n*self.anchor_len),j,i].view(1)
                self.obj += obj_pred
                obj_pred_list.append(obj_pred)
                #calculate the class loss
                classes_pred = x_b[(best_n*self.anchor_len + 7):((best_n+1)*self.anchor_len), j, i].view(self.classes)
                classes_truth = Variable(torch.zeros(self.classes))
                classes_truth[class_truth] = 1
                if x.is_cuda:
                    classes_truth = classes_truth.cuda()
                classes_truth_list.append(classes_truth.view(1,self.classes))
                classes_pred_list.append(classes_pred.view(1,self.classes))
                self.class_precision += classes_pred[class_truth]
                if classes_pred[class_truth] > 0.5:
                    self.precision += 1
                #use for statistic 
                self.count += 1

        noobj_truth = torch.cat(truth_noobj_list, 0)
        noobj_pred = torch.cat(noobj_pred_list, 0)
        noobj_truth = noobj_truth.detach()
        self.loss_noobj = mse(noobj_pred, noobj_truth) * self.noobject_scale
        if self.seen < 500:
            box_t = Variable(torch.Tensor([0.5,0.5,0,0])).expand(in_height*in_width, 4).expand(self.num*batch_size, in_height*in_width, 4)
            if x.is_cuda:
                box_t = box_t.cuda()
            coords_pred = torch.cat(coords_pred_list, 0)
            box_t = box_t.detach()
            init_loss_coords = mse(coords_pred, box_t) * 0.01
        else:
            init_loss_coords = zeros

        pred_box = torch.cat(box_pred_list, 0)
        truth_box = torch.cat(box_truth_list, 0)
        pred_obj = torch.cat(obj_pred_list, 0)
        truth_obj = torch.ones(self.count)
        if x.is_cuda:
            truth_obj = truth_obj.cuda()
        
        pred_fy = torch.cat(euler_pred_list, 0)
        truth_fy = torch.cat(euler_truth_list, 0)
        truth_fy = truth_fy.detach()
        self.loss_fy = mse(pred_fy, truth_fy) * self.coord_scale 

        pred_classes = torch.cat(classes_pred_list, 0)
        truth_classes = torch.cat(classes_truth_list, 0)
        truth_box = truth_box.detach()
        self.loss_coords = mse(pred_box, truth_box) * self.coord_scale + init_loss_coords

        truth_obj = truth_obj.detach()
        self.loss_obj = mse(pred_obj, truth_obj) * self.object_scale
        truth_classes = truth_classes.detach()
        self.loss_classes = mse(pred_classes, truth_classes) * self.class_scale
        if self.count != 0:
            self.loss = self.loss_obj + self.loss_noobj + self.loss_coords + self.loss_classes +self.loss_fy
            self.Arecall = self.recall/self.count
            self.Aiou = self.iou/self.count
            self.AclassP = self.class_precision/self.count
            self.Aobj = self.obj/self.count
            self.Anoobj = self.noobj/ (in_height * in_width * self.num * batch_size)
        print('loss: %f, average IoU: %f, class: %f, Obj: %f, No obj: %f,  Recall: %f, count: %3d'%(self.loss,self.Aiou \
                ,self.AclassP,self.Aobj,self.Anoobj,self.Arecall,self.count) )
        print('loss_obj=%f, loss_noobj=%f, loss_coords=%f, loss_fy=%f, loss_classes=%f, loss=%f'%(self.loss_obj, self.loss_noobj, self.loss_coords,self.loss_fy, self.loss_classes, self.loss, ) )
        return self.loss
