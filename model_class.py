import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = nn.Conv2d(1, 32, 3, padding=1)
        self.acv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.con2 = nn.Conv2d(32, 64, 3, padding=1)
        self.acv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.con3 = nn.Conv2d(64, 32, 3, padding=1)
        self.acv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.con4 = nn.Conv2d(32, 16, 3, padding=1)
        self.acv4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(16*4*4, 512)
        self.act5 = nn.ReLU()
        self.lin2 = nn.Linear(512, 37)

    def forward(self, img):
        out = self.pool1(self.acv1(self.con1(img)))
        out = self.pool2(self.acv2(self.con2(out)))
        out = self.pool3(self.acv3(self.con3(out)))
        out = self.pool4(self.acv4(self.con4(out)))
        out = out.view(-1, 16*4*4)
        out = self.act5(self.lin1(out))
        out = self.lin2(out)

        return out