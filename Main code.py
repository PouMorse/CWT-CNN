# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:02:04 2024

@author: pmeshkiz
"""

"""
Assigning working directories and file names
"""
DataDirectory = r'C:\Users\pmeshkiz\OneDrive - Arizona State University (1)\ISNDE\Projects\AM parts(Two cubes)\Data\10' 
DataFile = r'\Interface.dat'
InfoFile = r'\Interface.dat.txt'

"""
Extracting inspection information
"""
from TecscanData import Extract_Inspection_Data, BinaryData_Reconstruction
import numpy as np

# Extracting Sacn points as ScanP, Index points as IndexP, and Sample per waveform as SamPW.
ScanP, IndexP, SamPW, samplingFreq = Extract_Inspection_Data(DataDirectory, InfoFile)

# Binary data extraction and reconstruction
UTData = BinaryData_Reconstruction(DataDirectory, DataFile, InfoFile)


"""
Plot C-scans
"""
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.express as px

# # A-scan for finding informative area
# plt.figure(figsize=[20,15])
# plt.plot(UTData[100,150,:])
# plt.ylabel('Amplitude')
# plt.xlabel('Time (microsecond)')
# plt.title('The A-scan of the center of Bump1 (for slicing)')
# plt.show()

# The best representation of bumps
BumpsDemo = UTData[:,:,5800:7000].max(axis=2)

# C-scan
fig1 = px.imshow(BumpsDemo.T) #Interactive C-scan to find the coordinates of bumps
fig1.show()


"""
Bumps Data extraction and A-scans
"""
# These data will be used for training and testing
Bump1 = UTData[85:115,125:175,5000:8000]
Bump2 = UTData[215:245,125:175,5000:8000]
Bump3 = UTData[335:365,125:175,5000:8000]


SamplingRange = 4000/samplingFreq + np.linspace(0, Bump1.shape[2]/samplingFreq,Bump1.shape[2])

# A-scan Bump1
plt.figure(figsize=[15,10])
plt.plot(SamplingRange*1e6,Bump1[15,25,:])
plt.ylabel('Amplitude', fontsize=18)
plt.xlabel('Time (microsecond)', fontsize=18)
plt.title('The A-scan of the center of Bump1', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# A-scan Bump2
plt.figure(figsize=[15,10])
plt.plot(SamplingRange*1e6,Bump2[15,25,:])
plt.ylabel('Amplitude', fontsize=18)
plt.xlabel('Time (microsecond)', fontsize=18)
plt.title('The A-scan of the center of Bump2', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# A-scan Bump3
plt.figure(figsize=[15,10])
plt.plot(SamplingRange*1e6,Bump3[15,25,:])
plt.ylabel('Amplitude', fontsize=18)
plt.xlabel('Time (microsecond)', fontsize=18)
plt.title('The A-scan of the center of Bump3', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

"""
FFT for demonstration
"""
# Peaking the same signals as their A-scan is demonstrated above and padding them
FBump1 = np.pad(Bump1[15,25,:],(0, UTData.shape[2]-Bump1.shape[2]),'constant')
FBump2 = np.pad(Bump2[15,25,:],(0, UTData.shape[2]-Bump2.shape[2]),'constant')
FBump3 = np.pad(Bump3[15,25,:],(0, UTData.shape[2]-Bump3.shape[2]),'constant')

# Amplitude of FFT
fft_bump1 = np.abs(np.fft.fft(FBump1))
fft_bump2 = np.abs(np.fft.fft(FBump2))
fft_bump3 = np.abs(np.fft.fft(FBump3))

# Extract Fruequencies in MHz
FFT_freqs = (np.fft.fftfreq(len(fft_bump1), d=1.0/samplingFreq))/1e6


# Dominant Frquencies
DFrequ_Bump1 = FFT_freqs[np.argmax(fft_bump1[:int(len(FBump1)/2)])]
DFrequ_Bump2 = FFT_freqs[np.argmax(fft_bump2[:int(len(FBump1)/2)])]
DFrequ_Bump3 = FFT_freqs[np.argmax(fft_bump3[:int(len(FBump1)/2)])]

print(f"The Dominant Frequency in the signal received from"
      f" the center of Bump1 is {DFrequ_Bump1:.3f} MHz")
print(f"The Dominant Frequency in the signal received from"
      f" the center of Bump2 is {DFrequ_Bump2:.3f} MHz")
print(f"The Dominant Frequency in the signal received from"
      f" the center of Bump3 is {DFrequ_Bump3:.3f} MHz")

# Plot FFT data

plt.figure(figsize=[15,10])
plt.plot(FFT_freqs[:int(len(FBump1)/2)],fft_bump1[:int(len(FBump1)/2)])
plt.xlabel('Frequency (MHz)', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.title('FFT of the Signal form the center of Bump1', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.show()


plt.figure(figsize=[15,10])
plt.plot(FFT_freqs[:int(len(FBump1)/2)],fft_bump2[:int(len(FBump1)/2)])
plt.xlabel('Frequency (MHz)', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.title('FFT of the Signal form the center of Bump2', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.show()


plt.figure(figsize=[15,10])
plt.plot(FFT_freqs[:int(len(FBump1)/2)],fft_bump3[:int(len(FBump1)/2)])
plt.xlabel('Frequency (MHz)', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.title('FFT of the Signal form the center of Bump3', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.show()



"""
CWT for demonstration
"""
from WaveletAnalysis import CWT

wavelet = 'cmorl1.5-0.5'
scales = np.arange(1, 100) # Goes down to almost 600 KHz

Temp1 = CWT(Bump1[15, 25, :], wavelet, scales, samplingFreq, display=True)
Temp2 = CWT(Bump2[15, 25, :], wavelet, scales, samplingFreq, display=True)
Temp3 = CWT(Bump3[15, 25, :], wavelet, scales, samplingFreq, display=True)

"""
CWT for CNN
"""
# Assign the length and height of the CWT images
imHeight = len(scales)
imLength =  Bump1.shape[2]
numClasses = 3 # refers to the number bumps

# Initializing storage matrices
cwt_Bump1 = CWT(Bump1, wavelet, scales, samplingFreq)
cwt_Bump2 = CWT(Bump2, wavelet, scales, samplingFreq)
cwt_Bump3 = CWT(Bump3, wavelet, scales, samplingFreq)


"""
Defining CNN model (MineI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda')
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class MineI(nn.Module):
    def __init__(self, numClasses, device = 'cuda'):
        super(MineI, self).__init__()
        self.Convs = nn.Sequential(
            nn.Conv2d(1,8, kernel_size=(1,20), padding=1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8,64, kernel_size=(1,20), padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64,128, kernel_size=(1,20), padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.dropout = nn.Dropout(0.5)
        self.Dens1 = None
        self.Dens2 = None
        self.Dens3 = None
        self.numClasses = numClasses
        
    def Sizing_Est(self, InPUt):
        SIZE = InPUt.size()[1] * InPUt.size()[2] * InPUt.size()[3]
        if self.Dens1 is None and self.Dens2 is None:
            self.Dens1 = nn.Linear(SIZE,128).to(device)
            self.Dens2 = nn.Linear(128,32).to(device)
            self.Dens3 = nn.Linear(32,self.numClasses).to(device)
        return SIZE
        
    def forward (self, x):
        x = self.Convs(x)
        x = x.view(-1, self.Sizing_Est(x)) # flatten the data
        x = F.relu(self.Dens1(x))
        x = self.dropout(x)
        x = F.relu(self.Dens2(x))
        x = self.dropout(x)
        x = self.Dens3(x)
        y = F.log_softmax(x, dim=1)
        return y
        
"""
Calling model and Otimizers
"""

model = MineI(numClasses, device=device).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)
LossFn = nn.CrossEntropyLoss()

"""
Data Preprocssing
"""

# Flattening Data
cwt_Bump1_prepross = cwt_Bump1.reshape(-1, imHeight, imLength)
cwt_Bump1_prepross = (torch.tensor(cwt_Bump1_prepross, dtype=torch.float32,
                                   device=device)).unsqueeze(1)
cwt_Bump2_prepross = cwt_Bump2.reshape(-1, imHeight, imLength)
cwt_Bump2_prepross = (torch.tensor(cwt_Bump2_prepross, dtype=torch.float32,
                                   device=device)).unsqueeze(1)
cwt_Bump3_prepross = cwt_Bump3.reshape(-1, imHeight, imLength)
cwt_Bump3_prepross = (torch.tensor(cwt_Bump3_prepross, dtype=torch.float32,
                                   device=device)).unsqueeze(1)

# Creating labels
Bump1_label = torch.zeros(cwt_Bump1_prepross.shape[0], dtype=torch.long)
Bump2_label = torch.ones(cwt_Bump2_prepross.shape[0], dtype=torch.long)        
Bump3_label = 2 * torch.ones(cwt_Bump3_prepross.shape[0], dtype=torch.long)     

# Concatenating data and labels
data = torch.cat([cwt_Bump1_prepross, cwt_Bump2_prepross, cwt_Bump3_prepross], dim=0)
labels = torch.cat([Bump1_label, Bump2_label, Bump3_label], dim=0)      

# Splitting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
Train_set = TensorDataset(X_train, y_train)    
Test_set = TensorDataset(X_test, y_test)

# Formatting training and test batches
Train_loader = DataLoader(Train_set, batch_size=50, shuffle=True)
Test_loader = DataLoader(Test_set, batch_size=10, shuffle=False) 


"""
Training the model MineI
"""

numEpochs = 20 # The value is low due to batch training
loss_plot_ite = [] #for plotting the trianing loss per iteration(SGD)
loss_plot_ave = [] # for plotting the trianing loss per epoch
Val_loss_plot = [] # for plotting validation loss per epoch

for Epoch in range(numEpochs):

    loss_acc = 0 # Give total loss at the end of each epoch (for monitoring)
    for x_value, y_value in Train_loader:
        x_value = x_value.to(device)
        y_value = y_value.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        yh = model(x_value)
        loss = LossFn(yh, y_value)
        loss.backward()
        optimizer.step()
        
        loss_plot_ite.append(loss.item())
        loss_acc += loss.item()
        
    print(f'Epoch {Epoch + 1}, Loss: {loss_acc / len(Train_loader)}')
    loss_plot_ave.append(loss_acc/len(Train_loader))
    
    #Perform validation on training set
    Val_Correct = 0
    Val_Accuracy = []
    Val_loss_acc = 0
    with torch.no_grad():
        for xt_value, yt_value in Train_loader:
            xt_value = xt_value.to(device)
            yt_value = yt_value.to(device)
            
            yth = model(xt_value)
            validation_loss = LossFn(yh, y_value)
            Val_loss_acc += validation_loss.item()
            USELESS, Prediction = torch.max(yth.data,1)
            Val_Correct += (Prediction == yt_value).sum().item()
            
        Val_Accuracy.append(Val_Correct/y_train.size()[0])
        Val_loss_plot.append(Val_loss_acc/len(Train_loader))
        
# Plot SGD demonstration    
plt.figure(figsize=[10, 8])
plt.plot(list(range(len(loss_plot_ite))), loss_plot_ite, label='Training Loss') 
plt.title("Demonstration of SGD")
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.legend()
plt.show()

# Plot loss behavior
plt.figure(figsize=[10, 8])
plt.plot(list(range(1, numEpochs + 1)), loss_plot_ave, label='Training Loss')
plt.plot(list(range(1, numEpochs + 1)), Val_loss_plot, label='Validation Loss') 
plt.title("Loss over Epochs")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show() 

# Plot Validatin accuracy
plt.figure(figsize=[10, 8])
plt.plot(list(range(1, numEpochs + 1)), Val_loss_plot) 
plt.title("Model Accuracy on training data")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show() 
    
"""
Model Evaluation
"""
Correct = 0
Accuracy = 0

with torch.no_grad():
    for xt_value, yt_value in Test_loader:
        xt_value = xt_value.to(device)
        yt_value = yt_value.to(device)
        
        yth = model(xt_value)
        USELESS, Prediction = torch.max(yth.data,1)
        Correct += (Prediction == yt_value).sum().item()
        
    Accuracy = Correct/y_test.size()[0]

# print(f"The Number of corrected answers: {Correct}")
print(f"The accuracy of the model is {Accuracy}")
