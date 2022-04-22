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
from torch.utils.data.sampler import SubsetRandomSampler

# from unet import UNet
from AttRCNN_UNet import Att_R2U
# from auto import encoder, decoder

from collections import OrderedDict
# =============================================================

# Realtime Graph
# =============================================================
# Bokeh
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from functools import partial
from threading import Thread
from tornado import gen
from torch.autograd import Variable

# =============================================================
import wandb

wandb.login()

torch.cuda.empty_cache()

wandb.init(project="Doa1_PreTraining_50000_ALL_2")
# Loading Dataset  
# =============================================================


max_r = 0.7422917323053082
max_i = 0.763537867946671
max_p = 3.141592533629258
min_r = -0.6849425146673929
min_i = -0.7019437502051241
min_p = -3.1415916362555287

df = sio.loadmat("../datasets/doa2/SNR_NS_ALL_50000_2.mat")


class DOA_dataset(Dataset):
    def __init__(self):
        transp = np.transpose(df['NS_data'], (2, 0, 1))
        new = np.zeros((300000, 3, 8, 100))
        for i in range(0, transp.shape[0]):
            for j in range(0, transp.shape[1]):
                for k in range(0, transp.shape[2]):
                    new[i][0][j][k] = (transp[i][j][k].real - min_r)/(max_r-min_r)
                    new[i][1][j][k] = (transp[i][j][k].imag - min_i)/(max_i-min_i)
                    new[i][2][j][k] = (cmath.phase(transp[i][j][k]) - min_p)/(max_p-min_p)

        self.x = torch.from_numpy(new)
        self.y = torch.from_numpy(np.asarray(df['DOA']))
        self.n_sample = len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_sample

dataset = DOA_dataset()
validation_split = .1
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
print(dataset_size)

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

train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=512,
                                                sampler=valid_sampler)

# =============================================================

# =============================================================

# Model 
unet = Att_R2U()

optimizer = optim.Adam(unet.parameters(), lr=0.0001)

criterion = nn.BCELoss()

# Using GPU if available 
if torch.cuda.is_available():
	print(torch.cuda.get_device_name(0))
	unet = unet.cuda()
	criterion = criterion.cuda()

# exp_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma = 0.1)
def train():
  ''' Funtion for training  '''

  for i in range(200):
    # Dataset Training Loss
    unet.train()
    training_loss = 0
    for features, labels in train_loader:
      inputs, labels = Variable(features.cuda()), Variable(labels.cuda())
      optimizer.zero_grad()
      output = unet(inputs.float().cuda())
      output = torch.sigmoid(output)
      losss = criterion(output, inputs.float().cuda())
      losss.backward()
      optimizer.step()
      
      # exp_scheduler.step()
      training_loss += losss.item()

    # Dataset Validation
    unet.eval()
    validation_loss = 0
    with torch.no_grad():
      for features, labels in validation_loader:
        inputs, labels = Variable(features.cuda()), Variable(labels.cuda())
        output = unet(inputs.float().cuda())
        output = torch.sigmoid(output)
        loss = criterion(output, inputs.float().cuda())
        validation_loss += loss.item()
    wandb.log({"Training loss":(training_loss/len(train_loader)), "Validation loss":( validation_loss/len(validation_loader))})
    print("Epoch {} - Traningloss: {}".format(i+1, training_loss/len(train_loader)))
    print("Validationloss: {}".format( validation_loss/len(validation_loader)))
  wandb.finish()

  # Saving Model as .pth file
  torch.save(unet.state_dict(), "./SNR_50000_dropout.pth")
  print("Training complete, model weights are saved")

# =============================================================
# Realtime Graph 
train()













