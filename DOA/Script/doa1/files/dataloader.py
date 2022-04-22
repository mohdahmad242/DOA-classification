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
# def norm(file):
#   logging.info('Calculating Maximum and minimum values')
#   df = file
#   # print(df["DOA"].shape)
#   # print(df['NS_data'].shape)
#   transp = np.transpose(df['NS_data'], (2, 0, 1))
#   new = np.zeros((50000, 3, 8, 100))
#   for i in range(0, transp.shape[0]):
#       for j in range(0, transp.shape[1]):
#           for k in range(0, transp.shape[2]):
#               new[i][0][j][k] = transp[i][j][k].real
#               new[i][1][j][k] = transp[i][j][k].imag
#               new[i][2][j][k] = cmath.phase(transp[i][j][k])

#   max_r =  -10000000000000
#   min_r = 10000000000000
#   max_i =  -10000000000000
#   min_i = 10000000000000
#   max_p =  -10000000000000
#   min_p = 10000000000000

#   for i in range(0, new.shape[0]):
#       for j in range(0, new.shape[1]):
#           for k in range(0, new.shape[2]):
#             if new[i][0][j][k] > max_r :
#               max_r = new[i][0][j][k]
#             if new[i][0][j][k] < min_r:
#               min_r = new[i][0][j][k]
#             if new[i][1][j][k] > max_i :
#               max_i = new[i][1][j][k]
#             if new[i][1][j][k] < min_i:
#               min_i = new[i][1][j][k]
#             if new[i][2][j][k] > max_p :
#               max_p = new[i][2][j][k]
#             if new[i][2][j][k] < min_p:
#               min_p = new[i][2][j][k]


#   print(max_r)
#   print(max_i)
#   print(max_p)
#   print(min_r)
#   print(min_i)
#   print(min_p)
#   return max_r, max_i, max_p, min_r, min_i, min_p

df1  = sio.loadmat("../datasets/doa1/SNR_NS_0_50000_1.mat")
# df2  = sio.loadmat("../datasets/doa1/SNR_NS_10_50000_1.mat")
# df3  = sio.loadmat("../datasets/doa1/SNR_NS_20_50000_1.mat")
# df4  = sio.loadmat("../datasets/doa1/SNR_NS_30_50000_1.mat")
# df5  = sio.loadmat("../datasets/doa1/SNR_NS_40_50000_1.mat")
# df6  = sio.loadmat("../datasets/doa1/SNR_NS_50_50000_1.mat")
# df = pd.DataFrame(np.hstack((df1['NS_data'], df1['DOA'])))

norm(df1)
norm(df2)
norm(df3)
norm(df4)
norm(df5)
norm(df6)
# print(df.info())
# # # print(df1["DOA"].shape)
# # # print(df1['NS_data'].shape)
# new_dict = {
# 	"NS_data": np.concatenate([df1["NS_data"], df2["NS_data"],df3["NS_data"],df4["NS_data"],df5["NS_data"],df6["NS_data"]],axis=2),
# 	"DOA": np.concatenate([df1["DOA"], df2["DOA"],df3["DOA"],df4["DOA"],df5["DOA"],df6["DOA"]])
# }

# sio.savemat('../datasets/doa2/SNR_NS_ALL_50000_2.mat', new_dict)

train_dataset_list = []
test_dataset_list = []

def get_data(df, max_r, max_i, max_p, min_r, min_i, min_p):
  class DOA_dataset(Dataset):
    def __init__(self, df):
        transp = np.transpose(df['NS_data'], (2, 0, 1))
        new = np.zeros((50000, 3, 8, 100))
        for i in range(0, transp.shape[0]):
            for j in range(0, transp.shape[1]):
                for k in range(0, transp.shape[2]):
                    new[i][0][j][k] = (transp[i][j][k].real - min_r)/(max_r-min_r)
                    new[i][1][j][k] = (transp[i][j][k].imag - min_i)/(max_i-min_i)
                    new[i][2][j][k] = (cmath.phase(transp[i][j][k]) - min_p)/(max_p-min_p)

        self.x = torch.from_numpy(new)
        self.y = torch.from_numpy(np.asarray(df['DOA']))
        self.n_sample = len(self.y)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_sample


  dataset = DOA_dataset(df)

  # print(df1['DOA'], "Label")
  # print(len(df1['DOA']))
  validation_split = 0.1
  shuffle_dataset = True
  random_seed= 42

  # Creating data indices for training and validation splits:
  dataset_size =  len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(validation_split * dataset_size))
  if shuffle_dataset :
      np.random.seed(random_seed)
      np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]


  # Creating PT data samplers and loaders:
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)

  #dataloader = DataLoader(dataset=dff, batch_size=100, shuffle=True,  num_workers=2)

  train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, 
                                             sampler=train_sampler)
  validation_loader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                                  sampler=valid_sampler)

  train_dataset_list.append(train_loader)
  test_dataset_list.append(validation_loader)

