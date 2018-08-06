# -*- coding: utf-8 -*-
'''
这个文件是用来将雷达点云bin文件绘制成png图片的
'''
import os 
import os.path
import numpy as np
import cv2
import math
#需转换的文件路径
file_path = "/home/zft/velodyne/file/"
files = os.listdir(file_path)
#保存图片的路径
img_path = "/home/zft/velodyne/img/"
#newbin_path = "/home/zft/velodyne/newbin/"
#雷达扫射的最远距离
range_l = 40
#雷达扫射的最大宽度范围
width = 80
#循环读取文件
for binfile in files:
    filename = os.path.split(binfile)[-1].split('.')[0]#获取文件名
    f = file_path + binfile
    a = np.fromfile(f,dtype=np.float32)#获取二进制文件信息
    col = 4
    row = int(len(a)/col)
    a = np.reshape(a,(row,col))#排成512行1024列，每4个数为一行
    r=np.zeros((512,1024))
    g=np.zeros((512,1024))
    b=np.zeros((512,1024))
    #循环读取一行
    for i in range(row):
        if 0<a[i][0]<range_l and -(width/2)<a[i][1]<(width/2) and -2<a[i][2]<1.25:#获取符合条件的数据点
            n = 1023-int((a[i][1]+(width/2))*1024/width)#将雷达坐标转为图像坐标
            m = 511-int(a[i][0]*512/range_l)
            r[m][n] =r[m][n] + 1
            if g[m][n] < a[i][2]:
                g[m][n] = a[i][2]
            if b[m][n] < a[i][3]:
                b[m][n] = a[i][3]
    for i in range(512):
        for j in range(1024):
            r[i][j]=np.log10(r[i][j]+1)/64
            if r[i][j]>1.0:
                r[i][j]=1.0
    #数据归一化后转为RGB像素值
    g_max = np.max(g)
    g_min = np.min(g)
    scale = 1/(g_max - g_min)
    bin_g = (g - g_min)*scale
    g = bin_g*255
    g = g.astype(np.uint8)
    
    b_max = np.max(b)
    b_min = np.min(b)
    scale = 1/(b_max - b_min)
    bin_b = (b - b_min)*scale
    b = bin_b*255
    b = b.astype(np.uint8)

    r_max = np.max(r)
    r_min = np.min(r)
    scale = 1/(r_max - r_min)
    bin_r = (r - r_min)*scale
    r = bin_r*255
    r = r.astype(np.uint8)
   
    merged = cv2.merge([r,g,b])#三通道合并为一张图片
    cv2.imwrite(img_path + filename +".png" ,merged)

   # filesave = open(os.path.join(newbin_path + filename + '.bin'),'wb')
   # bin_r.tofile(filesave)
   # bin_g.tofile(filesave)
   # bin_b.tofile(filesave)
   # filesave.close()
    
    print(filename + ".bin has convered.")

    #cv2.waitKey(0)
print("finished!")






