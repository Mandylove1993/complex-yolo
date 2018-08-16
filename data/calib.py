#这个是将label文件，也就是归一化后的雷达坐标系下的图像坐标转化为相机坐标系下的图像坐标，画3D框在原图上
import os 
import os.path
import math
import numpy as np
import pandas as pd
import cv2
import linecache

width = 80
range_l = 80

labelfile = "/home/zft/velodyne/label/000002.txt"
calibfile = "/home/zft/velodyne/calib/000002.txt" 
img = cv2.imread('/home/zft/velodyne/img/000002.png')
image = cv2.imread('/home/zft/velodyne/images/000002.png')
lines_calib = open(calibfile).readlines()
lines_label = open(labelfile).readlines()
#计算P2
count_p2 = linecache.getline(calibfile,3)
count_p2 = count_p2.replace('\n',' ').replace(':',' ').split(' ')
df_p2 = pd.DataFrame({'P2':count_p2[0],'data':count_p2[2:14:1]})#将list转为表格
list_p2 = df_p2['data'].apply(pd.to_numeric)#将str转为数值类型
P2 = np.array(list_p2).reshape(3,4)
print(P2)
#计算R 
count_r = linecache.getline(calibfile,5)
count_r = count_r.replace('\n',' ').replace(':',' ').split(' ')
df_r = pd.DataFrame({'R0_rect':count_r[0],'data':count_r[2:11:1]})
list_r = df_r['data'].apply(pd.to_numeric)
r_1 = np.array(list_r).reshape(3,3)
r_1 = np.pad(r_1,pad_width=((0,1),(0,1)),mode='constant',constant_values=((0,0),(0,0)))
r_1[3,3] = 1
R = r_1
print(R)
#计算T 
count_t = linecache.getline(calibfile,6)
count_t = count_t.replace('\n',' ').replace(':',' ').split(' ')
df_t = pd.DataFrame({'Tr_velo_to_cam':count_t[0],'data':count_t[2:14:1]})
list_t = df_t['data'].apply(pd.to_numeric)
t_1 = np.array(list_t).reshape(3,4)
t_1 = np.pad(t_1,pad_width=((0,1),(0,0)),mode='constant',constant_values=((0,0),(0,0)))
t_1[3,3] = 1
T = t_1
print(T)
#矩阵相乘
C = np.zeros((4,4))
C = np.dot(np.dot(P2,R),T)

for line in lines_label:
    l = line.replace('\n',' ').split(' ')
    df = pd.DataFrame({'class':l[0],'data':l[1:8:1]})
    lc = df['data'].apply(pd.to_numeric)
    #获取中心点的雷达图像坐标系
    xi = lc[0]*1024
    yi = lc[1]*1024
    wi = lc[2]*1024
    li = lc[3]*1024
    im = lc[5]
    re = lc[6]
    Li = np.sqrt(math.pow(wi,2) + math.pow(li,2))
    print(xi, yi, wi, li, Li)
    fy = math.atan2(im,re)
    alpa = math.atan2(wi,li)
    #雷达坐标系下8顶点的图像坐标
    xi1 = xi + Li/2*math.cos(alpa + fy)
    yi1 = yi - Li/2*math.sin(alpa + fy)
    xi2 = xi - Li/2*math.cos(fy - alpa)
    yi2 = yi + Li/2*math.sin(fy - alpa)
    xi3 = xi - Li/2*math.cos(alpa + fy)
    yi3 = yi + Li/2*math.sin(alpa + fy)
    xi4 = xi + Li/2*math.cos(fy - alpa)
    yi4 = yi - Li/2*math.sin(fy - alpa)
    #雷达坐标
    xv1 = range_l - yi1/1024*range_l
    yv1 = width/2 - xi1/1024*width 
    xv2 = range_l - yi2/1024*range_l
    yv2 = width/2 - xi2/1024*width
    xv3 = range_l - yi3/1024*range_l
    yv3 = width/2 - xi3/1024*width
    xv4 = range_l - yi4/1024*range_l
    yv4 = width/2 - xi4/1024*width
    #获取最大高度
    left_x = int(min(xi1,xi2,xi3,xi4))
    left_y = int(min(yi1,yi2,yi3,yi4))
    right_x = int(max(xi1,xi2,xi3,xi4))
    right_y = int(max(yi1,yi2,yi3,yi4))
    cropImg = img[left_x:right_x,left_y:right_y]
    b,g,r = cv2.split(cropImg)
    h = np.max(g)/255*3.25+2
    '''
    b,g,r = cv2.split(img)
    H = np.zeros((1024,1024))
    for i in range(1024):
        for j in range(1024):
            if  int(xi - Li/2) < j < int(xi + Li/2) and int(yi - Li/2)< i <int(yi + Li/2) :
                H[i][j] = g[i][j]
            else:
                continue
    h = np.max(H)/255*3.25+2
    print(h)
    '''
    z0 = h - 1.73 #=zv5,zv6,zv7,zv8
    zh = -1.73  #=zv1,zv2,zv3,zv4
    zc1 = xv1 - 0.27
    zc2 = xv2 - 0.27
    zc3 = xv3 - 0.27
    zc4 = xv4 - 0.27
    #雷达坐标系到相机坐标系
    #x = - y'
    #y = - (z' + 0.08)
    #z = x' - 0.27
    xc1 = - yv1
    yc1 = - (zh + 0.08)
    xc2 = - yv2
    yc2 = - (zh + 0.08)
    xc3 = - yv3
    yc3 = - (zh + 0.08)
    xc4 = - yv4
    yc4 = - (zh + 0.08)
    xc5 = - yv1
    yc5 = - (z0 + 0.08)
    xc6 = - yv2
    yc6 = - (z0 + 0.08)
    xc7 = - yv3
    yc7 = - (z0 + 0.08)
    xc8 = - yv4
    yc8 = - (z0 + 0.08)
    #包装成点改变维度
    pc1 = (xc1,yc1,zc1,1)
    pc1 = np.reshape(pc1,(4,1))
    pc2 = (xc2,yc2,zc2,1)
    pc2 = np.reshape(pc2,(4,1))
    pc3 = (xc3,yc3,zc3,1)
    pc3 = np.reshape(pc3,(4,1))
    pc4 = (xc4,yc4,zc4,1)
    pc4 = np.reshape(pc4,(4,1))
    pc5 = (xc5,yc5,zc1,1)
    pc5 = np.reshape(pc5,(4,1))
    pc6 = (xc6,yc6,zc2,1)
    pc6 = np.reshape(pc6,(4,1))
    pc7 = (xc7,yc7,zc3,1)
    pc7 = np.reshape(pc7,(4,1))
    pc8 = (xc8,yc8,zc4,1)
    pc8 = np.reshape(pc8,(4,1))

    '''
    #包装成点改变维度
    pv1 = (xv1,yv1,zh,1)
    pv1 = np.reshape(pv1,(4,1))
    pv2 = (xv2,yv2,zh,1)
    pv2 = np.reshape(pv2,(4,1))
    pv3 = (xv3,yv3,zh,1)
    pv3 = np.reshape(pv3,(4,1))
    pv4 = (xv4,yv4,zh,1)
    pv4 = np.reshape(pv4,(4,1))
    pv5 = (xv1,yv1,z0,1)
    pv5 = np.reshape(pv5,(4,1))
    pv6 = (xv2,yv2,z0,1)
    pv6 = np.reshape(pv6,(4,1))
    pv7 = (xv3,yv3,z0,1)
    pv7 = np.reshape(pv7,(4,1))
    pv8 = (xv4,yv4,z0,1)
    pv8 = np.reshape(pv8,(4,1))
    

    #变换矩阵，将雷达坐标转换为相机下的图像坐标系
    img_p1 = np.dot(C,pv1)/zc1
    img_p2 = np.dot(C,pv2)/zc2
    img_p3 = np.dot(C,pv3)/zc3
    img_p4 = np.dot(C,pv4)/zc4
    img_p5 = np.dot(C,pv5)/zc1
    img_p6 = np.dot(C,pv6)/zc2
    img_p7 = np.dot(C,pv7)/zc3
    img_p8 = np.dot(C,pv8)/zc4

    '''
    #将相机坐标系转化为图像坐标系
    img_p1 = np.dot(P2,pc1)/zc1
    img_p2 = np.dot(P2,pc2)/zc2
    img_p3 = np.dot(P2,pc3)/zc3
    img_p4 = np.dot(P2,pc4)/zc4
    img_p5 = np.dot(P2,pc5)/zc1
    img_p6 = np.dot(P2,pc6)/zc2
    img_p7 = np.dot(P2,pc7)/zc3
    img_p8 = np.dot(P2,pc8)/zc4


    #去掉最后一行
    point1 = np.reshape(img_p1,(1,3))
    point1 = np.delete(point1,2,1)
    point1 = point1.flatten()
    point1 = point1.astype(np.int16)
    point1 = tuple(point1)
    point2 = np.reshape(img_p2,(1,3))
    point2 = np.delete(point2,2,1)
    point2 = point2.flatten()
    point2 = point2.astype(np.int16)
    point2 = tuple(point2)
    point3 = np.reshape(img_p3,(1,3))
    point3 = np.delete(point3,2,1)
    point3 = point3.flatten()
    point3 = point3.astype(np.int16)
    point3 = tuple(point3)
    point4 = np.reshape(img_p4,(1,3))
    point4 = np.delete(point4,2,1)
    point4 = point4.flatten()
    point4 = point4.astype(np.int16)
    point4 = tuple(point4)
    point5 = np.reshape(img_p5,(1,3))
    point5 = np.delete(point5,2,1)
    point5 = point5.flatten()
    point5 = point5.astype(np.int16)
    point5 = tuple(point5)
    point6 = np.reshape(img_p6,(1,3))
    point6 = np.delete(point6,2,1)
    point6 = point6.flatten()
    point6 = point6.astype(np.int16)
    point6 = tuple(point6)
    point7 = np.reshape(img_p7,(1,3))
    point7= np.delete(point7,2,1)
    point7 = point7.flatten()
    point7 = point7.astype(np.int16)
    point7 = tuple(point7)
    point8 = np.reshape(img_p8,(1,3))
    point8 = np.delete(point8,2,1)
    point8 = point8.flatten()
    point8 = point8.astype(np.int16)
    point8 = tuple(point8)

    print(point1, point2, point3, point4, point5, point6, point7, point8)

    #画图
    img2 = cv2.line(image,point1,point2,(255,255,0))
    img2 = cv2.line(image,point2,point3,(255,255,0))
    img2 = cv2.line(image,point3,point4,(255,255,0))
    img2 = cv2.line(image,point4,point1,(255,255,0))
    img2 = cv2.line(image,point5,point6,(255,255,0))
    img2 = cv2.line(image,point6,point7,(255,255,0))
    img2 = cv2.line(image,point7,point8,(255,255,0))
    img2 = cv2.line(image,point8,point5,(255,255,0))
    img2 = cv2.line(image,point1,point5,(255,255,0))
    img2 = cv2.line(image,point2,point6,(255,255,0))
    img2 = cv2.line(image,point3,point7,(255,255,0))
    img2 = cv2.line(image,point4,point8,(255,255,0))

cv2.imshow("polt",img2)
cv2.waitKey(0)






