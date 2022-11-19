#!/usr/bin/env python
# coding: utf-8

# In[18]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[19]:


import matplotlib.pyplot as plt


# In[20]:


train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')
print(train_data.shape)
print(test_data.shape)


# In[21]:


train_data


# In[22]:


X_train = train_data.iloc[:,1:].values.reshape(-1, 1, 28, 28)
y_train = train_data.iloc[:,:1].values.flatten()
X_test = test_data.values.reshape(-1, 1, 28, 28)


# In[23]:


plt.imshow(X_train[5].reshape(28, 28, 1));


# In[24]:


y_train[5]


# In[25]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[26]:


class MnistData(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# In[27]:


X_train = torch.tensor(X_train, dtype=torch.float32) / 255
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test, dtype=torch.float32) / 255


# In[28]:


train_set = MnistData(X_train, y_train)
test_set = MnistData(X_test)


# In[29]:


train_loader = DataLoader(train_set, batch_size=512)
test_loader = DataLoader(test_set, batch_size=512)


# In[30]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1600, 10)
        
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.flatten(1)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    
    # This is not a typical way to create a train function
    # I like to embed into the model class itself
    def fit(self, train_loader, epochs=1):
        # Turn to train mode
        self.train()
        for epoch in range(epochs):
            # Get loss and accuracy per epoch
            total_loss = 0
            total_acc = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.cuda()
                target = target.cuda()
                # Zero grad
                self.optimizer.zero_grad()
                # Get the output
                output = self(data)
                # Calculate the loss (It is average by default)
                loss = F.nll_loss(output, target)
                # Do gradient math
                loss.backward()
                self.optimizer.step()
                # Calculate accuracy and sum over batches
                acc = (output.argmax(1) == target).cpu().numpy().mean()
                total_loss += loss.item()
                total_acc += acc
            
            # Averaging over batch or it will be wrong
            total_loss = total_loss / (batch_idx+1)
            total_acc = total_acc / (batch_idx+1)
            print(f"Train Epoch: {epoch}\tLoss: {total_loss:.6f}\tAccuracy: {total_acc:.6f}")


# In[31]:


model = CNN().cuda()


# In[33]:


model.fit(train_loader, epochs=200)


# In[ ]:
for batch_idx, (data) in enumerate(test_loader):
    data = data.cuda()
    
    test_outputs = model(data)
    _, predicted = torch.max(test_outputs, 1)
    
    df = pd.DataFrame(predicted.cpu().numpy())
    if (0 == batch_idx):
        results = df
    else:
        results = results.append(df, ignore_index=True)
    
results.insert(results.shape[1], 'ImageId', results.index+1)
results.columns = ['Label', 'ImageId']

order = ['ImageId', 'Label']
results = results[order]

results.to_csv('submission.csv', index=None)


