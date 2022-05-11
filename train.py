import os
from tkinter import Variable
import numpy as np
from sklearn.cluster import KMeans
import segmentation_models as sm
from data.dataloader import CityScapesDataset
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from net import UNet
from utils import LoadImage, CreateKMeans, CheckAccuracy
from losses import total_loss_fn

# Initializaing UNet architecture and clustering
net = UNet(3, 15).float() # Using float precision
km = CreateKMeans(num_clusters=15)

criterion = nn.CrossEntropyLoss()
sm_loss = sm.losses.CategoricalFocalLoss()
total_loss = total_loss_fn()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_fn = total_loss


train_data = CityScapesDataset(km = km,
                img_dir='data/cityscapes/train')
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

val_data = CityScapesDataset(km = km,
                img_dir='data/cityscapes/val')
val_loader = DataLoader(val_data, batch_size=8, shuffle=True)

# for i, batch in enumerate(train_loader):
#     print(i)

num_epochs = 1
for epoch in range(num_epochs):  # loop over the dataset multiple times
    logs = {}
    total_correct = 0
    total_loss = 0
    total_images = 0
    total_val_loss = 0
    running_loss = 0.0

    CheckAccuracy(val_loader, net, 15)

    net.train()
    for i, (img, seg) in (pbar := tqdm(enumerate(train_loader))):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        img = img.permute(0, 3, 1, 2).float()
        seg = seg.permute(0, 3, 1, 2).float()
        outputs = net(img)
        loss = torch.Tensor(np.array(loss_fn(outputs.detach().numpy(), seg.detach().numpy()))).float()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

        # # Obtaining predictions from max value
        # _, predicted = torch.max(outputs.detach(), 1)
        # total_images+= seg.size(0)

        # # Calculate the number of correct answers
        # correct = (predicted == seg).sum().item()

        # total_correct+=correct
        # total_loss+=loss.item()

        # print statistics
        running_loss += loss.item()
        if i % 2 == 0:    # print every 2000 mini-batches
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2:.3f}')
            # progress bar
            pbar.set_description(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2:.3f}')
            running_loss = 0.0

    CheckAccuracy(val_loader, net, 15)

print('Finished Training')
