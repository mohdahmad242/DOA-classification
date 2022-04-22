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

#from unet import UNet
from classifier import encoder

from collections import OrderedDict
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
#==========================================================================
# For Plotting loss graph
# Bokeh
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from functools import partial
from threading import Thread
from tornado import gen
from AttRCNN_UNet import Att_R2U
from sklearn.metrics import precision_score
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# import sys

# values = sys.argv

# datastePath = values[1]
# projectName = values[2]

# logging.info("File name - "+datastePath +" Started Training")
import wandb

wandb.login()

torch.cuda.empty_cache()

wandb.init(project="SNR_NS_50_150000_2")
#==========================================================================

df1 = sio.loadmat("/home/iiitd/Desktop/Ahmad/datasets/SNR_NS_50_150000_2.mat")

from dataloader import norm
# =============================================================

# max_r, max_i, max_p, min_r, min_i, min_p = norm(df1)
max_r = 0.3380636549028141
max_i = 0.3352641191613967
max_p = 3.1415909745354496
min_r = -0.34773794934191427
min_i = -0.33667363233086717
min_p = -3.141592538319071

class DOA_dataset(Dataset):
    def __init__(self, df):
        transp = np.transpose(df['NS_data'], (2, 0, 1))
        new = np.zeros((150000, 3, 8, 100))
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


dataset = DOA_dataset(df1)

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

train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=1024,
                                                sampler=valid_sampler)
#==========================================================================

#==========================================================================
# preTrained = Att_R2U()


# pre_model = torch.load("/home/iiitd/Desktop/Ahmad/doa2/SNR_50000_dropout.pth")

# preTrained.load_state_dict(pre_model, strict=False)

# for prams in preTrained.parameters():
#   prams.requires_grad = False
en = encoder()
# de = decoder()
autoencoder = nn.Sequential( en)
#==========================================================================



criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
	print(torch.cuda.get_device_name(0))
	autoencoder = autoencoder.cuda()
	optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4, weight_decay=1e-5)
	criterion = criterion.cuda()

# print(autoencoder)
#==========================================================================
def train():
	for i in range(200):

		training_loss = 0
		tcorrect = 0
		ttotal = 0
		autoencoder.train()
		for features, labels in train_loader:
			features, labels = Variable(features.cuda()), Variable(labels.cuda())
			optimizer.zero_grad()
			#unet_outputs = unet_model(features.float())
			enn = autoencoder(features.float())
			#auto_outputs = de_model(enn)
			auto_outputs = torch.transpose(enn, 2, 3)
			# print(auto_outputs.size())

			auto_outputs = torch.reshape(auto_outputs.cuda(), (auto_outputs.shape[0], 181, 2))
			
			losss = criterion(auto_outputs.cuda(), labels.type(torch.LongTensor).cuda())
			losss.backward()
			optimizer.step()
			training_loss += losss.item()

			_, pred = torch.max(auto_outputs, 1)
			ttotal+= labels.reshape(-1).size(0)

			tcorrect+=(pred.reshape(-1).cuda() == labels.reshape(-1)).sum().item()

		validation_loss = 0
		correct = 0
		total = 0
		autoencoder.eval()
		with torch.no_grad():
			for features, labels in validation_loader:
				features, labels = Variable(features.cuda()), Variable(labels.cuda())

				enn = autoencoder(features.float())
				auto_outputs = torch.transpose(enn, 2, 3)
				auto_outputs = torch.reshape(auto_outputs, (auto_outputs.shape[0], 181, 2))
				loss = criterion(auto_outputs.cuda(), labels.type(torch.LongTensor).cuda())

				_, pred = torch.max(auto_outputs, 1)
				total+= labels.reshape(-1).size(0)
				correct+=(pred.reshape(-1).cuda() == labels.reshape(-1)).sum().item()
				validation_loss += loss.item()
		# print(correct, "Correct")
		# print(total, "Total")
		wandb.log({"Trainig Acc":(100*(tcorrect/ttotal)), "Traningloss":(training_loss/len(train_loader)), 
			"Validation Acc":(100*(correct/total)), "Validationloss":( validation_loss/len(validation_loader))})
		# print("Epoch {} - Traningloss: {}".format(i+1, training_loss/len(train_loader)))
		# print("Validationloss: {}".format( validation_loss/len(validation_loader)))
		# print("Trainig Acc: {}".format( 100*(tcorrect/ttotal)))
		# print("Validation Acc: {}".format(100*(correct/total)))
	wandb.finish()
		

train()
logging.info("Training Complete")









