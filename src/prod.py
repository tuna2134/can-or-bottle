import torch
from torch.utils.data import Dataset
import torchvision

from PIL import Image
import torch.optim as optim
import torch.nn as nn
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
        # out: 222, 111
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        # out: 109, 54
        self.conv3 = nn.Conv2d(32, 64, 3, padding=0)
        # out: 52, 26
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(26 * 26 * 64, 128)
        self.fc2 = nn.Linear(128, 1)
        # self.sigmoid = nn.Sigmoid()
        # self.fc3 = nn.Linear(2, 1)

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
        # out = self.sigmoid(out)
        #  out = out.view(-1, 1)
        return out


trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
# trainset = torchvision.datasets.MNIST(root = 'path', train = True, download = True, transform = trans)
trainset = MyDataset(trans)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

device = torch.device("cpu")
net = CNNModel()
net = net.to(device)
criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

num_epochs = 50
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = net(data)
        target = target.float().unsqueeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()
        accuracy = correct / data.size(0)

        if batch_idx % 64 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(trainloader)}] Loss: {loss.item()} Accuracy: {accuracy:.4f}"
            )

## save onnx
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    net,
    dummy_input,
    "model.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
