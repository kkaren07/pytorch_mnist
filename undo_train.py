import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Conv2d(32,(3,3), kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32,(3,3), kernel_size=3, padding=1),
            nn.ReLu(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def __init__(self):
        model = nn.Sequential()
        model.add(Conv2d)
        model.add_module('fc1', nn.Linear(10,100))
        model.add_module('relu', nn.ReLU())
        model.add_module('fc2', nn.Linear(100,10))

net = Net()
