import torch
from torchsummary import summary
import torch.nn as nn
from torch.nn.modules import batchnorm
class BasicCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(BasicCNN,self).__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels= 16, kernel_size= (3,3),stride=1,padding=0)

        self.conv2 = nn.Sequential(
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3)),
        nn.ReLU(),
        nn.MaxPool2d((2,2)))
        self.avg = nn.Flatten()
        self.conv3 = nn.Sequential(
        nn.Linear(in_features=123008,out_features=32),
        nn.Linear(in_features=32,out_features=self.out_channels)
        )

        self.model2 = nn.Sequential(
           # nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=(4,4),padding=0),
           # nn.ReLU(),
           # nn.MaxPool2d((4,4)),
           # nn.Dropout(0.2),
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(4,4), padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(146880,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,self.out_channels),
        )
    def forward(self,x):
        return self.model2(x)
   
# CNN = BasicCNN(input_channels=1,output_channels=10)
# print(summary(CNN,(1,1025,40)))