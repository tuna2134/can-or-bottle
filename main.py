import torch
import torchvision

import numpy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 16, 3)
        # out: 26, 13
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        # out: 11, 5
        self.conv3 = nn.Conv2d(32, 64, 3, padding=0)
        # out: 3, 1
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(1 * 1 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
trainset = torchvision.datasets.MNIST(
    root="path", train=True, download=True, transform=trans
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=2
)

device = torch.device("cpu")
net = CNNModel()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)


num_epochs = 25
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(trainloader)}] Loss: {loss.item()}"
            )
