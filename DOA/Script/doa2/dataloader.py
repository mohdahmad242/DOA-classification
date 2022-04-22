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

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# # print(df.keys())
# print(df['DOA'].shape)
# print(df['NS_data'].shape)
# c = []
# for a in df['DOA']:
# 	c.append(a[0])


# d = pd.DataFrame(c)
# # ak = d.groupby(d[0]).count()
# # print(ak)
# aa = d[0].value_counts()
# print(aa)

# df1  = sio.loadmat("../datasets/doa2/SNR_NS_0_50000_2.mat")
# df2  = sio.loadmat("../datasets/doa2/SNR_NS_10_50000_2.mat")
# df3  = sio.loadmat("../datasets/doa2/SNR_NS_20_50000_2.mat")
# df4  = sio.loadmat("../datasets/doa2/SNR_NS_30_50000_2.mat")
# df5  = sio.loadmat("../datasets/doa2/SNR_NS_40_50000_2.mat")
# df6  = sio.loadmat("../datasets/doa2/SNR_NS_50_50000_2.mat")
# # df6  = sio.loadmat("../datasets/doa1/SNR_NS_0_50000_1.mat")
# # # df7  = sio.loadmat("../../DOA_dataset/Dataset_13_01_2021/SNR_NS_35.mat")
# # # df8  = sio.loadmat("../../DOA_dataset/Dataset_13_01_2021/SNR_NS_40.mat")
# # # df9  = sio.loadmat("../../DOA_dataset/Dataset_13_01_2021/SNR_NS_45.mat")
# # # df10 = sio.loadmat("../../DOA_dataset/Dataset_13_01_2021/SNR_NS_50.mat")

# # # print(df1["DOA"].shape)
# # # print(df1['NS_data'].shape)
# new_dict = {
# 	"NS_data": np.concatenate([df1["NS_data"], df2["NS_data"],df3["NS_data"],df4["NS_data"],df5["NS_data"],df6["NS_data"]],axis=2),
# 	"DOA": np.concatenate([df1["DOA"], df2["DOA"],df3["DOA"],df4["DOA"],df5["DOA"],df6["DOA"]])
# }

# sio.savemat('../datasets/doa2/SNR_NS_ALL_50000_2.mat', new_dict)

def norm(file):
  logging.info('Calculating Maximum and minimum values')
  df  = sio.loadmat(file)
  # print(df["DOA"].shape)
  # print(df['NS_data'].shape)
  transp = np.transpose(df['NS_data'], (2, 0, 1))
  new = np.zeros((100000, 3, 8, 100))
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
              max_i = new[i][1][j][k]
            if new[i][1][j][k] < min_i:
              min_i = new[i][1][j][k]
            if new[i][2][j][k] > max_p :
              max_p = new[i][2][j][k]
            if new[i][2][j][k] < min_p:
              min_p = new[i][2][j][k]

  print(file)          
  print(max_r)
  print(max_i)
  print(max_p)
  print(min_r)
  print(min_i)
  print(min_p)
  return max_r, max_i, max_p, min_r, min_i, min_p

import glob

files = glob.glob("../datasets/doa2_100k/*.mat")
for i in files:
  norm(i)