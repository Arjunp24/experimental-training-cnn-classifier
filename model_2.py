import torch.nn as nn
import torch.nn.functional as F


'''
Targets: Adjust model parameters to <8k 
Results:
    Parameters: 7828
    Best train accuracy: 99.11%
    Best test accuracy: 99.29%
Analysis: Remove dropout as there are a smaller number of parameters so having dropout will further reduce the number of trainable parameters 
File Link: https://github.com/Arjunp24/experimental-training-cnn-classifier/blob/main/model_2.py 
'''
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 10, kernel_size=1, padding=0)

        self.conv4 = nn.Conv2d(10, 16, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(16)

        self.avgpool = nn.AvgPool2d(6)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm2d(16)

        self.conv8 = nn.Conv2d(16, 10, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = F.max_pool2d(self.conv3(x), 2)
        x = self.dropout(F.relu(self.bn3(self.conv4(x))))
        x = self.dropout(F.relu(self.bn4(self.conv5(x))))
        x = self.dropout(F.relu(self.bn5(self.conv6(x))))
        x = self.conv8(self.avgpool(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
