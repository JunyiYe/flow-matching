import torch
import torch.nn as nn


def add_condition(tensor, cond_emb):
    cond_emb = cond_emb.view(cond_emb.size(0), cond_emb.size(1), 1, 1)
    return torch.cat([tensor, cond_emb.expand(-1, -1, tensor.size(2), tensor.size(3))], dim=1)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, condition_size):
        super().__init__()
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels + condition_size, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, condition):
        x = self.downconv(add_condition(x, condition))
        return x, self.maxpool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, condition_size):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels + condition_size, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.upconv = nn.Sequential(
            nn.Conv2d(in_channels + condition_size, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, residual_x, condition):
        x = self.deconv(add_condition(x, condition))
        x = torch.cat([x, residual_x], dim=1)
        x = self.upconv(add_condition(x, condition))
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.label_emb = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.t_emb = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.condition_size = 32

        self.down0 = DownSample(1, 64, self.condition_size)
        self.down1 = DownSample(64, 128, self.condition_size)
        self.down2 = DownSample(128, 256, self.condition_size)

        self.up0 = UpSample(256, 128, self.condition_size)
        self.up1 = UpSample(128, 64, self.condition_size)

        self.output_conv = nn.Conv2d(64 + self.condition_size, 1, kernel_size=3, padding=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, t, label):
        cond_emb = torch.cat([self.label_emb(label), self.t_emb(t.unsqueeze(1))], dim=1)
        x0, x = self.down0(x, cond_emb)
        x1, x = self.down1(x, cond_emb)
        x, _ = self.down2(x, cond_emb)
        x = self.up0(x, x1, cond_emb)
        x = self.up1(x, x0, cond_emb)
        return self.output_conv(add_condition(x, cond_emb))
