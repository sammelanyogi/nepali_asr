import torch.nn as nn
from torch.nn import functional as F

class HancyModel(nn.Module):
  def __init__(self):
    super(HancyModel, self).__init__()
    self.cnn = nn.Conv1d(128, 128, 10) 
    self.dense = nn.Linear(392, 128)
    self.lstm = nn.LSTM(128, 128, 1, bidirectional = False)
    self.dense2 = nn.Linear(128, 72)
  def forward(self, x):
    x = x.squeeze(1)
    x = self.cnn(x)
    x = self.dense(x)
    x = x.transpose(0,1)
    x, (hn, cn) = self.lstm(x)
    x = self.dense2(x)
    return x
