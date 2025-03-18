import torch
import torch.nn as nn

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        self.random_seed = 42

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout2d(0.1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU()
        self.drop3 = nn.Dropout2d(0)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.LeakyReLU()
        self.drop4 = nn.Dropout2d(0)
        self.pool2 = nn.MaxPool2d(2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        torch.manual_seed(self.random_seed)
        x = self.drop1(self.act1(self.bn1(self.conv1(x))))
        torch.manual_seed(self.random_seed)
        x = self.pool1(self.drop2(self.act2(self.bn2(self.conv2(x)))))
        torch.manual_seed(self.random_seed)
        x = self.drop3(self.act3(self.bn3(self.conv3(x))))
        torch.manual_seed(self.random_seed)
        x = self.pool2(self.drop4(self.act4(self.bn4(self.conv4(x)))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    res =CIFAR10CNN()
    t = torch.randn((10, 3, 224, 224))
    t = res(t)
    print(t.shape)
