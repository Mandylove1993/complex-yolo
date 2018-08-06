'''
这个文件是用来将相机坐标点txt文件转换为训练网络所需的label标签的
'''
import os 
import os.path
import math
import numpy as np
import pandas as pd
import cv2

width = 80
range_l = 40
#文件打开和保存路径
file_path = "/home/zft/velodyne/label_test/"
save_path = '/home/zft/velodyne/labels/'
files = os.listdir(file_path)
#循环读取文件
for labelfile in files:
    #保存数据的空数组
    bx = []
    by = []
    bw = []
    bl = []
    bc = []
    bf = []
    bm = []
    be = []
    filename = os.path.split(labelfile)[-1].split('.')[0]    
    lines = open(os.path.join(file_path + labelfile)).readlines()
    for line in lines:
        line = line.replace('\n', '')
        l = line.split(' ')
        if l[0] == "DontCare" :
            continue
        else:
            class_name = l[0]#将class转为对应的0,1,...7的8个分类标志
            if class_name == 'Car':
                class_name = '0'
            elif class_name == 'Van':
                class_name = '1'
            elif class_name == 'Truck':
                class_name = '2'
            elif class_name == 'Pedestrian':
                class_name = '3'
            elif class_name == 'Person_sitting':
                class_name = '4'
            elif class_name == 'Cyclist':
                class_name = '5'
            elif class_name == 'Tram':
                class_name = '6'
            else:
                class_name = '7'
            df = pd.DataFrame({'class':l[0],'data':l[1:15:1]})
            l_new = df['data'].apply(pd.to_numeric)#将str转为数值型
            #获取相机坐标系
            x = l_new[10]
            y = l_new[11]
            z = l_new[12]
            w = l_new[8]
            le = l_new[9]
            ry = l_new[13]
            fy = - ry
            #将相机坐标系转为雷达坐标系
            v_x = z + 0.27
            v_y = - x 
            if 0 < z <(range_l-0.27) and  -range_l < x < range_l and -1.25 < y <2 :#在相机坐标系下选择符合条件的点，并计算归一化值
                bc.append(class_name)
                im_y = (1023-int((v_y + (width/2))*1024/width))/1024
                by.append(im_y)
                im_x = (511-int(v_x * 512/range_l))/512
                bx.append(im_x)
                im_w = w/512
                bw.append(im_w)
                im_l = le/1024
                bl.append(im_l)
                im_fy = fy
                bf.append(im_fy)
                lm = math.sin(im_fy)
                bm.append(lm)
                re = math.cos(im_fy)
                be.append(re)  
            else:
                continue
    #打开循环保存数据的文件
    save_file = open(os.path.join(save_path + filename + '.txt'),'w')
    unuse_file = open("/home/zft/velodyne/labels/unuse_label.txt",'a+')#空label的文件保存路径
    size = len(bc)#获取文件行数
    if size == 0 :#若文件为空，则保存文件名到unuse_file
        print(filename +' ' + "no class")
        unuse_file.write(filename +'\n')
    else:#若文件不为空，则一行行保存数据，顺序为class,x,y,w,l,fy,im,ie
        for index in range(size):
            temp = bc[index] + ' ' + str(bx[index]) + ' ' + str(by[index]) + ' '+ str(bw[index]) + ' ' + str(bl[index]) + ' ' + str(bf[index]) + ' ' + str(bm[index]) + ' ' + str(be[index])
            save_file.write(temp + '\n')
save_file.close()
unuse_file.close()


       
        
       