import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio


transform = transforms.ToTensor()

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()
print(device)



class ConvAutoencoder(nn.Module):
    def __init__(self, img_ch=1,output_ch=1):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(1, 16, 2, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
              
        return x


#Instantiate the model
model = ConvAutoencoder()


#Loss function
criterion = nn.BCELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



df = sio.loadmat("../DOA_dataset/ULA_Code_for_DL/SNR_0.mat")
label = df['DOA']  
n = np.zeros((2000, 4, 180))

for i in range(0, 2000):
	for a in range(0, 4):
		n[i][a][label[i][a]-1] = 1


class DOA_dataset(Dataset):
    def __init__(self):
        transp = np.transpose(df['SNS_data'], (2, 0, 1))
        new = np.zeros((2000, 8, 100))
        for i in range(0, transp.shape[0]):
            for j in range(0, transp.shape[1]):
                for k in range(0, transp.shape[2]):
                    new[i][j][k] = (transp[i][j][k].real)

        

        self.x = torch.from_numpy(np.expand_dims(new, axis=1))
        self.y = torch.from_numpy(np.asarray(np.expand_dims(n, axis=1)))
        self.n_sample = 2000
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_sample

dff = DOA_dataset()
dataloader = DataLoader(dataset=dff, batch_size=100, shuffle=True,  num_workers=2)




#Epochs
n_epochs = 20

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training
    for features, labels in dataloader:

        optimizer.zero_grad()
        outputs = model(features.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*features.size(0)
          
    train_loss = train_loss/len(dataloader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


# for i in range(20):

# 	running_loss = 0

# 	for features, labels in dataloader:

# 		optimizer.zero_grad()

# 		output = model(features.float())
# 		losss = loss(output, labels)

# 		losss.backward()

# 		optimizer.step()

# 		running_loss += losss.item()

# 	else:
# 		print("Epoch {} - Traningloss: {}".format(i+1, running_loss/len(dataloader)))