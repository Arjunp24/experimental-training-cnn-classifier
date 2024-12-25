import torch.nn as nn
import torch.nn.functional as F

'''
Targets: Train baseline (using model from previous assignment) using 15 epochs  
Results:
    Parameters: 18596 
    Best train accuracy: 99.34% 
    Best test accuracy: 99.49%  
Analysis: Reduce parameters 
File Link: https://github.com/Arjunp24/experimental-training-cnn-classifier/blob/main/model_1.py 
'''
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 10, kernel_size=1, padding=0)

        self.conv4 = nn.Conv2d(10, 16, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(32)

        self.avgpool = nn.AvgPool2d(6)

        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm2d(16)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(16)

        self.conv8 = nn.Conv2d(16, 10, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = F.max_pool2d(self.conv3(x), 2)
        x = self.dropout(F.relu(self.bn3(self.conv4(x))))
        x = self.dropout(F.relu(self.bn4(self.conv5(x))))
        x = self.dropout(F.relu(self.bn5(self.conv6(x))))
        x = self.dropout(F.relu(self.bn6(self.conv7(x))))
        x = self.conv8(self.avgpool(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
