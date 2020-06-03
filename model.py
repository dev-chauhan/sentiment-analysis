from PQG.misc.HybridCNNLong import HybridCNNLong as Encoder
import torch.nn as nn
import torch
from dataset import word_to_ix

def to_onehot(t, c):
    return torch.zeros(*t.size(), c, device=t.device).scatter_(-1, t.unsqueeze(-1), 1)

class PretrainedEncoder(nn.Module):

    def __init__(self, vocab_size, dropout=0.5):
        super(PretrainedEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, 512, dropout=dropout, avg=1)
        checkpoint = torch.load("pretrained/150_-1.tar", map_location=torch.device("cpu"))
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    
    def forward(self, x):
        return self.encoder(to_onehot(x, self.vocab_size))


class Classifier(nn.Module):

  def __init__(self, inp_dim, hid1, out_dim):
    super(Classifier, self).__init__()
    self.dense1 = nn.Linear(inp_dim, hid1)
    self.relu = nn.ReLU()
    self.dropout1 = nn.Dropout(p=0.5)
    self.dense2 = nn.Linear(hid1, out_dim)
  
  def forward(self, x):
    return self.dense2(self.dropout1(self.relu(self.dense1(x))))
