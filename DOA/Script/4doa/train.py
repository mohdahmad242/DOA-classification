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
from auto import encoder, decoder

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

torch.cuda.empty_cache()
import wandb
wandb.login()

torch.cuda.empty_cache()

wandb.init(project="Doa1_SNR_NS_15_30000_500_4")
#==========================================================================

df1 = sio.loadmat("./SNR_NS_15_30000_500_4.mat")
#==========================================================================

max_r = 0.10607383042549647
max_i = -0.01824013671187322
max_p = -0.016187517914981613
min_r = -0.10432316827280157
min_i = -0.0018810992664454278
min_p = -0.001881099266445427

class DOA_dataset(Dataset):
    def __init__(self, df):
        transp = np.transpose(df['NS_data'], (2, 0, 1))
        new = np.zeros((30000, 3, 80, 500))
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

# print(df1['DOA'], "Label")
# print(len(df1['DOA']))

validation_split = .1
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
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

train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                                sampler=valid_sampler)
#==========================================================================


#==========================================================================

en = encoder()

# pre_model = torch.load("./preTrained/unet_norm.pth")

# en.load_state_dict(pre_model, strict=False)
de_model = decoder()

autoencoder = nn.Sequential(en, de_model)

#==========================================================================



criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
	print(torch.cuda.get_device_name(0))
	autoencoder = autoencoder.cuda()
	optimizer = optim.AdamW(autoencoder.parameters(), lr=0.0001)
	criterion = criterion.cuda()


#==========================================================================
def train():
	for i in range(300):

		training_loss = 0
		autoencoder.train()
		tcorrect = 0
		ttotal = 0
		for features, labels in train_loader:
			features, labels = Variable(features.cuda()), Variable(labels.cuda())
			optimizer.zero_grad()
			#unet_outputs = unet_model(features.float())
			enn = autoencoder(features.float())
			#auto_outputs = de_model(enn)
			auto_outputs = torch.transpose(enn, 2, 3)

			auto_outputs = torch.reshape(auto_outputs.cuda(), (auto_outputs.shape[0], 181, 4))
			# print(auto_outputs.size())
			# print(labels.size(), "labels")
			losss = criterion(auto_outputs.cuda(), labels.type(torch.LongTensor).cuda())
			_, pred = torch.max(auto_outputs, 1)
			ttotal+= labels.reshape(-1).size(0)
			# print(pred[0], "pred")
			# print(labels[0])

			tcorrect+=(pred.reshape(-1).cuda() == labels.reshape(-1)).sum().item()
			losss.backward()

			optimizer.step()
			training_loss += losss.item()

		validation_loss = 0
		correct = 0
		total = 0
		autoencoder.eval()
		with torch.no_grad():
			for features, labels in validation_loader:
				features, labels = Variable(features.cuda()), Variable(labels.cuda())

				enn = autoencoder(features.float())
				auto_outputs = torch.transpose(enn, 2, 3)
				auto_outputs = torch.reshape(auto_outputs, (auto_outputs.shape[0], 181, 4))
				loss = criterion(auto_outputs.cuda(), labels.type(torch.LongTensor).cuda())
				_, pred = torch.max(auto_outputs, 1)
				total+= labels.reshape(-1).size(0)


				correct+=(pred.reshape(-1).cuda() == labels.reshape(-1)).sum().item()
				validation_loss += loss.item()
		wandb.log({"Trainig Acc":(100*(tcorrect/ttotal)), "Traning loss":(training_loss/len(train_loader)), "Validation Acc":(100*(correct/total)), "Validation loss":( validation_loss/len(validation_loader))})
		print("Epoch {} - Traningloss: {}".format(i+1, training_loss/len(train_loader)))
		print("Validationloss: {}".format( validation_loss/len(validation_loader)))
		print("Trianing Acc: {}".format( 100*(tcorrect/ttotal)))
		print("Validaion Acc: {}".format( 100*(correct/total)))
	wandb.finish()
train()
# thread = Thread(target=train)
# thread.start()
print("Training Complete")
#==========================================================================
# print(unet_model.state_dict())
#torch.save(unet_model.state_dict(), "./unet_wieghts.pth")









