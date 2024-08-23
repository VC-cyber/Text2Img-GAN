import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Convolutional layers with increasing channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4 + 512, 512)  # Combine image features with caption embedding
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x, caption_embedding):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        # Concatenate the image features with the caption embedding
        x = torch.cat((x, caption_embedding), dim=1)
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x
    
    def extract_features(self, x, caption_embedding):
    # """
    # Extract features from the last convolutional layer.
    # """
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        features = F.leaky_relu(self.conv3(x), 0.2)  # Extract features from this layer
        
        return features
