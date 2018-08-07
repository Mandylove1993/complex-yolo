'''
   这个文件是用来进行测试的，当获得雷达数据bin文件生成的图像文件后，用这个脚本将相机的txt文件所记录的物体坐标提取转换后变成图像坐标并绘制在生成的图像文件中，
   如何可以框对，那么就说明转换没有问题
'''
import os 
import os.path
import math
import numpy as np
import pandas as pd
import cv2

width = 80
range_l = 40

file = "/home/zft/velodyne/label_test/000004.txt"
lines = open(file).readlines()
img = cv2.imread('/home/zft/velodyne/img/000004.png')

for line in lines:
    line = line.replace('\n', '')
    l = line.split(' ')
    #忽略Dontcare
    if l[0] == "DontCare":
        continue
    else:
        df = pd.DataFrame({'class':l[0],'data':l[1:15:1]})#将list转为表格
        l = df['data'].apply(pd.to_numeric)#将str转为数值类型
        x = l[10]
        z = l[12]
        w = l[8]
        le = l[9]
        ry = l[13]
        fy =-ry
        alpa = math.atan(w/le)
        #计算矩形的四个顶点坐标
        q = np.sqrt(math.pow(w,2) + math.pow(le,2))/2
        p = fy - alpa
        k = fy + alpa
        x1 = q*math.sin(p) + z + 0.27
        y1 = -(q*math.cos(p) + x)
        x2 = z - q*math.sin(p)
        y2 = -(x - q*math.cos(p))
        x3 = q*math.sin(k) + z + 0.27
        y3 = -(q*math.cos(k) + x)
        x4 = z - q *math.sin(k)
        y4 = -(x - q*math.cos(k))
        x5 = z + 0.27
        y5 = - x
        #将雷达坐标转为图像坐标
        n1 = 1023-int((y1+(width/2))*1024/width)
        m1 = 511-int(x1*512/range_l)
        n2 = 1023-int((y2+(width/2))*1024/width)
        m2 = 511-int(x2*512/range_l)
        n3 = 1023-int((y3+(width/2))*1024/width)
        m3 = 511-int(x3*512/range_l)
        n4 = 1023-int((y4+(width/2))*1024/width)
        m4 = 511-int(x4*512/range_l)
        n5 = 1023-int((y5+(width/2))*1024/width)
        m5 = 511-int(x5*512/range_l)
        #四顶点连线
        img2 = cv2.line(img,(n1,m1),(n3,m3),(255,255,0))
        img2 = cv2.line(img,(n3,m3),(n2,m2),(255,255,0))
        img2 = cv2.line(img,(n2,m2),(n4,m4),(255,255,0))
        img2 = cv2.line(img,(n4,m4),(n1,m1),(255,255,0))
        img2 = cv2.line(img,(int((n1+n3)/2),int((m1+m3)/2)),(n5,m5),(255,255,0))
cv2.imshow("000007",img2)
cv2.waitKey(0)




        

               
             