import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNetLikeBlock(nn.Module):
    def __init__(self, channel):
        super(ResNetLikeBlock, self).__init__()
        self.feats = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
        )
    def forward(self, x):
        identity = x
        out = self.feats(x)
        out += identity
        assert(x.shape == self.feats(x).shape)
        return F.relu(out)

class Classifier(nn.Module):
    # TODO: implement me
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)

        self.block0 = ResNetLikeBlock(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.block1 = ResNetLikeBlock(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)

        self.block2 = ResNetLikeBlock(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)

        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block0(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.block1(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.block2(x)
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = sF.relu(elf.conv5(x))
        # print(x.shape)
        x = x.view(x.size()[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


