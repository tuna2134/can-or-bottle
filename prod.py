import torch
from torch.utils.data import Dataset
import torchvision

from PIL import Image
import numpy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


IMAGE_DIR = Path("images")


class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for i, dir in enumerate(sorted(IMAGE_DIR.iterdir())):
            for file in dir.iterdir():
                self.image_paths.append(file)
                self.labels.append(i)
    
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        # image = image.resize((28, 28))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, 3)
        # out: 126, 63
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        # out: 61, 30
        self.conv3 = nn.Conv2d(32, 64, 3, padding=0)
        # out: 28, 14
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(14 * 14 * 64, 128)
        self.fc2 = nn.Linear(128, 2)
    
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

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
# trainset = torchvision.datasets.MNIST(root = 'path', train = True, download = True, transform = trans)
trainset = MyDataset(trans)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True)

device = torch.device("cpu")
net = CNNModel()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)


num_epochs = 100
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # predict
        # preds = output.argmax(dim=1, keepdim=True)
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()
        accuracy = correct / data.size(0)

        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(trainloader)}] Loss: {loss.item()} Accuracy: {accuracy:.4f}')

## save onnx
dummy_input = torch.randn(1, 3, 128, 128)
torch.onnx.export(net, dummy_input, "model.onnx", verbose=True)