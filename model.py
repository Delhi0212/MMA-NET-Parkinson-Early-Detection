import torch
import torch.nn as nn

class MMANet(nn.Module):
    def __init__(self):
        super(MMANet, self).__init__()

        # Voice branch
        self.voice_fc = nn.Sequential(
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Gait branch
        self.gait_lstm = nn.LSTM(input_size=16, hidden_size=64, batch_first=True)

        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, voice, gait):
        v = self.voice_fc(voice)

        _, (h, _) = self.gait_lstm(gait)
        g = h[-1]

        combined = torch.cat((v, g), dim=1)
        out = self.fc(combined)

        return out
