import torch
from torch import nn

class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM1, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc =  nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self,x):
        h_0 = (torch.zeros(self.num_layers, x.shape[0], self.hidden_size)).cuda()
        c_0 = (torch.zeros(self.num_layers, x.shape[0], self.hidden_size)).cuda()
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn[-1]
        out = self.fc(hn)
        out = self.relu(out)
        return out
