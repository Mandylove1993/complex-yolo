import os 
import os.path
import struct
import numpy as np
import cv2
import math

f = '/home/zft/velodyne/file/000001.bin'
range_l = 40
width = 80
a = np.fromfile(f,dtype=np.float32)
col = 4
row = int(len(a)/col)
a = np.reshape(a,(row,col))
r=np.zeros((512,1024))
g=np.zeros((512,1024))
b=np.zeros((512,1024))

for i in range(row):
    if 0<a[i][0]<range_l and -(width/2)<a[i][1]<(width/2) and -2<a[i][2]<1.25:
        n = 1023-int((a[i][1]+width/2)*1024/width)
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

g_max = g.max()
g_min = g.min()
scale = 1/(g_max - g_min)
bin_g = (g - g_min)*scale
g = bin_g*255
g = g.astype(np.uint8)

b_max = np.max(b)
b_min = np.min(b)
bin_b = (b - b_min)/(b_max - b_min)
b = bin_b*255
b = b.astype(np.uint8)

r_max = np.max(r)
r_min = np.min(r)
bin_r = (r - r_min)/(r_max - r_min)
r = bin_r*255
r = r.astype(np.uint8)

merged = cv2.merge([r,g,b])
cv2.imwrite("merge.png" ,merged)







