import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.init_size = 8  # Start with a larger initial size to reduce over-parameterization
        self.fc1 = nn.Sequential(
            nn.Linear(input_size + 512, 256 * self.init_size ** 2),  # Reduced dimensionality
            nn.BatchNorm1d(256 * self.init_size ** 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample from (8, 8) to (16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample from (16, 16) to (32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Final output (32, 32, 3)
            nn.Tanh()  # Output image with values in range [-1, 1]
        )

    def forward(self, x, caption_embedding):
        x = torch.cat((x, caption_embedding), dim=1)
        x = self.fc1(x)
        x = x.view(x.shape[0], 256, self.init_size, self.init_size)  # Reshape to (batch_size, 256, 8, 8)
        x = self.conv_blocks(x)
        return x
