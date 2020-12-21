import torch
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

class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
    
    def forward(self, x):
        x = x.transpose(1, 2)
        # x = self.norm(self.dropout(F.gelu(x)))
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x


class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 72, #total no. of characters in our dataset
        "n_feats": 81, #Total no. of features extracted
        "dropout": 0.1,
        "hidden_size": 1024,
        "num_layers": 1
    }

    def __init__(self, hidden_size = 1024, num_classes = 72, n_feats = 128, num_layers = 1, dropout = 0.1):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2),
            ActDropNormCNN1D(n_feats, dropout),
        )
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.0,
                            bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))

    def forward(self, x, hidden):
        x = x.squeeze(1)  # batch, feature, time
        x = self.cnn(x) # batch, time, feature
        x = self.dense(x) # batch, time, feature
        x = x.transpose(0, 1) # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)