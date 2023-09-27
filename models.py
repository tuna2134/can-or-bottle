import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(23.5 * 23.5 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
