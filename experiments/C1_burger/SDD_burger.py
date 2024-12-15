# %%

import os
import sys
sys.path.append("/home/user/Documents/GPO/general")
import gpytorch
from gpytorch.means import MultitaskMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from utilities3 import *
from pytorch_wavelets import DWT, IDWT
from pytorch_wavelets import DWT1D, IDWT1D
import scipy
from _sdd import *
from utils import *
# %%
'''  
DEVICE
'''
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

print(f"Device: {device}")
# torch.manual_seed(0)
# np.random.seed(0)


# %%
'''   
CONFIG AND DATA 
'''
ntrain = 1100
ntest = 200
# SDD parameters
lr = 0.01
momentum = 0.9
iterations = 5000
B = 4
polyak=1e-2#1e-2
noise_scale=0.02
length_scale = 13.6 

sub = 2**4  # subsampling rate
h = 2**13 // sub  
s = h


# loading the data
dataloader = MatReader(
   r"/home/user/Documents/GP_WNO/DATA/burgers_data_R10.mat")
x_data = dataloader.read_field("a")[:, ::sub]
y_data = dataloader.read_field("u")[:, ::sub]



x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)


x_tr = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
y_tr = y_train

# test data
x_t = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
y_t = y_test


x_tr=x_tr.to(device)
y_tr=y_tr.to(device)
x_t=x_t.to(device)
y_t=y_t.to(device)

#%%

Kernel = ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)).to(device=device)
Kernel.base_kernel.lengthscale= length_scale

sdd_algo = SDDAlgorithm(Kernel, x_tr, y_tr, iterations=iterations, polyak=polyak,B=B,device=device)
alpha_polyak_trained,_ = sdd_algo.train()

all_predictions = []
with torch.no_grad():
    for i in range(0, len(x_t)):
        x_batch = x_t[i:i+1]  
        predictions_batch = Kernel(x_batch, x_tr) @ alpha_polyak_trained  
        all_predictions.append(predictions_batch)  

y_pred_sdd = torch.cat(all_predictions, dim=0)


#%% 

'''PREDICTION ERROR'''
mse_loss = nn.MSELoss()
prediction_error = mse_loss(y_pred_sdd.to(device), y_t)

relative_error = torch.mean(torch.linalg.norm(y_pred_sdd.to(device)-y_t, axis = 1)/torch.linalg.norm(y_t, axis = 1))


print(f'MSE Testing error: {(prediction_error).item()}')
print(f'Mean relative error: {100*relative_error} % ')

#%%