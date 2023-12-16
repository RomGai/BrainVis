#Original model presented in: C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, Deep Learning Human Mind for Automated Visual Classification, CVPR 2017 
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np

class Model(nn.Module):

    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size,40)
        
    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x=x.cpu()
        fourier_transform = np.fft.fft(x, axis=2)
        half_spectrum = fourier_transform[:, :, 1:440 // 2 + 1]
        amplitude_spectrum = np.abs(half_spectrum)

        amplitude_spectrum = torch.tensor(amplitude_spectrum).float()

        x = amplitude_spectrum.permute(0, 2, 1)
        x = x.to("cuda")

        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        # Forward LSTM and get final state
        #print(np.shape(x))
        x = self.lstm(x, lstm_init)[0][:,-1,:]
        reps=x
        #print(np.shape(x))
        #reps=self.output(x)
        # Forward output
        xa = F.relu(self.output(x))
        # x=self.fc(xa)
        # x=F.relu(x)
        x = self.classifier(xa)
        return x,xa
