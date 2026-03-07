import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNLSTMModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv(x)

        x = x.permute(0,2,1)

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        out = self.fc(out)

        return out