import scipy.io as sio
import numpy as np
import torch
from torch import nn, optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math
import pandas as pd
import cmath
# transp = np.transpose(df['SNS_data'], (2, 0, 1))
# new = np.zeros((2000, 8, 100))
# for i in range(0, transp.shape[0]):
#     for j in range(0, transp.shape[1]):
#         for k in range(0, transp.shape[2]):
#             new[i][j][k] = (transp[i][j][k].real)


# label = df['DOA'] 
# print(label.shape) 

# image = torch.rand(100, 1, 4, 180)
# transp = torch.transpose(image, 2, 3)
# reudce = torch.reshape(transp, (100, 180, 4))

# print(image.shape)
# print(transp.shape)

# print(reudce.shape)

# # DOA
# print(df['DOA'][0])
# print(df['DOA'][0][0])
# label = []

# for i in range(0, 2000):
# 	label.append(df['DOA'][i][0])


# print(len(label))
# print(label)
# a = pd.get_dummies(df['DOA']
# print(a)

# d = {}
# for i in range(1, 181):
# 	d[i] = np.zeros(2000)

# new_df = pd.DataFrame(d)




# n = np.zeros((2000, 4, 180))

# for i in range(0, 2000):
# 	for a in range(0, 4):
# 		n[i][a][label[i][a]-1] = 1

# print(n[1])
df = sio.loadmat("./SNR_NS_15_30000_500_4.mat")
print(df.keys())

transp = np.transpose(df['NS_data'], (2, 0, 1))
new = np.zeros((30000, 3, 80, 500))
for i in range(0, transp.shape[0]):
    for j in range(0, transp.shape[1]):
        for k in range(0, transp.shape[2]):
            new[i][0][j][k] = transp[i][j][k].real
            new[i][1][j][k] = transp[i][j][k].imag
            new[i][2][j][k] = cmath.phase(transp[i][j][k])

max_r =  -10000000000000
min_r = 10000000000000
max_i =  -10000000000000
min_i = 10000000000000
max_p =  -10000000000000
min_p = 10000000000000

for i in range(0, new.shape[0]):
    for j in range(0, new.shape[1]):
        for k in range(0, new.shape[2]):
          if new[i][0][j][k] > max_r :
            max_r = new[i][0][j][k]
          if new[i][0][j][k] < min_r:
            min_r = new[i][0][j][k]
          if new[i][1][j][k] > max_i :
            max_i = new[i][0][j][k]
          if new[i][1][j][k] < min_i:
            min_i = new[i][0][j][k]
          if new[i][2][j][k] > max_p :
            max_p = new[i][0][j][k]
          if new[i][2][j][k] < min_p:
            min_p = new[i][0][j][k]


print(max_r)
print(max_i)
print(max_p)
print(min_r)
print(min_i)
print(min_p)

new1 = np.zeros((30000, 3, 80, 500))
for i in range(0, transp.shape[0]):
    for j in range(0, transp.shape[1]):
        for k in range(0, transp.shape[2]):
            new[i][0][j][k] = (transp[i][j][k].real - min_r)/(max_r-min_r)
            new[i][1][j][k] = (transp[i][j][k].imag - min_i)/(max_i-min_i)
            new[i][2][j][k] = (cmath.phase(transp[i][j][k]) - min_p)/(max_p-min_p)


sio.savemat('./SNR_NS_15_30000_500_4_norm.mat', new1)