import os
import random
from PIL import Image, ImageEnhance
from PIL import ImageFile
import torch
import numpy as np
import torchvision as tv 
from torch.utils import data
from torchvision import transforms as T 

ImageFile.LOAD_TRUNCATED_IMAGES = True
class YoloDataset(data.Dataset):
    def __init__(self, listFile, width, height, truth=1, data_expand=1, train=1):
        self.imageList = []
        self.labelList = []
        with open(listFile,'r') as fp:
            for line in fp.readlines():
                imageDir = line.replace('\n','')
                labelDir = imageDir.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
                self.imageList.append(imageDir)
                self.labelList.append(labelDir)
        self.len = self.imageList.__len__()
        self.data_expand = data_expand
        self.truth = truth
        self.train = train
        self.width = width
        self.height = height
        self.width_now = width
        self.height_now = height
        if data_expand:
            self.saturation = 1.5
            self.exposure = 1.5
            self.hue = 1.5
            self.sharpness = 1.5
        else:
            self.saturation = 0
            self.exposure = 0
            self.hue = 0
        self.seen = 0
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        self.seen += 1
        imageDir = self.imageList[index]
        labelDir = self.labelList[index]
        pil_img = Image.open(imageDir)
        mode = pil_img.mode
        while mode != 'RGB':
            index = int(random.random()*self.len)
            imageDir = self.imageList[index]
            labelDir = self.labelList[index]
            pil_img = Image.open(imageDir)
            mode = pil_img.mode
        transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        '''
        if self.train:
            
            if (self.seen)%1280 == 1 and self.seen > 6400:
                self.width_now = (random.randint(0,6) + 10)*32
                self.height_now = (random.randint(0,6) + 10)*32
                print('resizing input %d x %d'%(self.width_now,self.height_now))
            img = pil_img.resize( (self.width_now, self.height_now) )
            
            if self.data_expand:
                #Brightness,Color,Contrast,Sharpness range from 0.5~1.5
                #change exposure
                enh_bri = ImageEnhance.Brightness(img)
                brightness = self.exposure - random.random() 
                img = enh_bri.enhance(brightness)
                #change color
                enh_col = ImageEnhance.Color(img)
                color = self.saturation - random.random()
                img = enh_col.enhance(color)
                #change Contrast 
                enh_con = ImageEnhance.Contrast(img)
                Contrast = self.hue - random.random()
                img = enh_con.enhance(Contrast)
                #change Sharpness
                enh_sha = ImageEnhance.Sharpness(img)
                sharpness =  self.sharpness - random.random()
                img = enh_sha.enhance(sharpness)
                
        else:
            '''
        img = pil_img.resize( (self.width, self.height) )
        image = transform(img)
        label = torch.zeros(50,8)
        if self.train:
            objs = []
            with open(labelDir,'r') as fl:
                for line in fl.readlines():
                    obj = line.replace(' \n','').replace('\n','').split(' ')
                    objs.append(obj)
            for i in range(min(objs.__len__(), 50) ):
                #print(objs[i])
                assert(objs[i].__len__() == 8)
                label[i][0] = float(objs[i][1])
                label[i][1] = float(objs[i][2])
                label[i][2] = float(objs[i][3])
                label[i][3] = float(objs[i][4])
                label[i][4] = float(objs[i][5])
                label[i][5] = float(objs[i][6])
                label[i][6] = float(objs[i][7])
                label[i][7] = float(objs[i][0])
        return (image, label)
