import torch
import torch.nn as nn
import torch.nn.functional as F

class SincConv_fast(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(SincConv_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        return self.conv(x)

class RawNet2(nn.Module):
    def __init__(self):
        super(RawNet2, self).__init__()
        self.conv1 = SincConv_fast(128, 251)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(3)

        self.res_block1 = self._make_res_block(128, 128)
        self.res_block2 = self._make_res_block(128, 256)
        self.res_block3 = self._make_res_block(256, 512)

        self.gru = nn.GRU(input_size=512, hidden_size=1024, num_layers=1, batch_first=True)
        self.fc = nn.Linear(1024, 1)

    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # last time-step
        x = self.fc(x)
        return torch.sigmoid(x)
