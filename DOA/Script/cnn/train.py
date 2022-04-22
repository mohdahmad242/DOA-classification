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
from auto import encoder

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

wandb.init(project="Doa1_SNR_NS_30_100000_100_2_CNN")
#==========================================================================

df1 = sio.loadmat("../datasets/SNR_NS_30_100000_100_2.mat")
#==========================================================================

# ("../../DOA_dataset/Dataset_19_12_2020/SNR_NS_50.mat")
class DOA_dataset(Dataset):
    def __init__(self, df):
        transp = np.transpose(df['NS_data'], (2, 0, 1))
        new = np.zeros((100000, 3, 80, 100))
        for i in range(0, transp.shape[0]):
            for j in range(0, transp.shape[1]):
                for k in range(0, transp.shape[2]):
                    new[i][0][j][k] = transp[i][j][k].real
                    new[i][1][j][k] = transp[i][j][k].imag
                    new[i][2][j][k] = cmath.phase(transp[i][j][k])

        

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

train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                                sampler=valid_sampler)
#==========================================================================


#==========================================================================

autoencoder = encoder()

# pre_model = torch.load("./preTrained/unet_norm.pth")

# en.load_state_dict(pre_model, strict=False)
# de_model = decoder()

# autoencoder = nn.Sequential(en, de_model)


#==========================================================================



criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
	print(torch.cuda.get_device_name(0))
	autoencoder = autoencoder.cuda()
	optimizer = optim.AdamW(autoencoder.parameters(), lr=0.00001, weight_decay=1e-5)
	criterion = criterion.cuda()

ep = []
train_loss_l = []
test_loss_l = []
#==========================================================================
def train():
	for i in range(200):

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

			auto_outputs = torch.reshape(auto_outputs.cuda(), (auto_outputs.shape[0], 181, 2))
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
			for features, label in validation_loader:
				features, label = Variable(features.cuda()), Variable(label.cuda())

				enn = autoencoder(features.float())
				auto_outputs = torch.transpose(enn, 2, 3)
				auto_outputs = torch.reshape(auto_outputs, (auto_outputs.shape[0], 181, 2))
				
				loss = criterion(auto_outputs.cuda(), label.type(torch.LongTensor).cuda())
				_, pred = torch.max(auto_outputs, 1)
				total+= label.reshape(-1).size(0)
				# print(pred[0], "pred")
				# print(label[0])

				correct+=(pred.reshape(-1).cuda() == label.reshape(-1)).sum().item()
				validation_loss += loss.item()
		ep.append(i)
		train_loss_l.append(training_loss/len(train_loader))
		test_loss_l.append( validation_loss/len(validation_loader))
		wandb.log({"Trainig Acc":(100*(tcorrect/ttotal)), "Traning loss":(training_loss/len(train_loader)), "Validation Acc":(100*(correct/total)), "Validation loss":( validation_loss/len(validation_loader))})
		print("Epoch {} - Traningloss: {}".format(i+1, training_loss/len(train_loader)))
		print("Validationloss: {}".format( validation_loss/len(validation_loader)))
		print("Trianing Acc: {}".format( 100*(tcorrect/ttotal)))
		print("Validaion Acc: {}".format( 100*(correct/total)))
		# new_data = {'epochs': [i+1],'trainlosses': [training_loss],'vallosses': [validation_loss]}
		# doc.add_next_tick_callback(partial(update, new_data))	
	wandb.finish()
train()
# thread = Thread(target=train)
# thread.start()
print("Training Complete")
#==========================================================================
# print(unet_model.state_dict())
#torch.save(unet_model.state_dict(), "./unet_wieghts.pth")









