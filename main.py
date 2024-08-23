import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'GAN_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data'))

from preprocessing import COCODataset
from generator import Generator
from discriminator import Discriminator
from train import train

def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

def main(): 
   # torch.cuda.empty_cache() 

    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(
                brightness=0.1 * torch.rand(1).item(),
                contrast=0.1 * torch.rand(1).item(),
                saturation=0.1 * torch.rand(1).item(),
                hue=0.1 * torch.rand(1).item()),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Hyperparameters
    batch_size = 128
    lr = 0.000025
    dlr = 0.000025
    epochs = 60
    max_batch = 1001
    noise_size = 256
    presave = True
    # Directories
    img_dir = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/GAN_project/Data/coco/images/train2014'
    ann_file = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/GAN_project/Data/coco/annotations/annotations/captions_train2014.json'

    # Initialize dataset and dataloader
    print("Loading dataset...")
    #dataset = COCODataset(img_dir, ann_file, transform=transform)
    dataset = datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Initialize models
    print("Initializing models...")
    generator = Generator(noise_size)
    discriminator = Discriminator()

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=dlr, betas=(0.5, 0.999))

    # Train the GAN
    print("Training GAN...")
    train(epochs, dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, batch_max=max_batch, preSave=presave)
    print("Training complete.")

    # Save models
    torch.save(generator.state_dict(), 'GAN_model/weights/generator.pth')
    torch.save(discriminator.state_dict(), 'GAN_model/weights/discriminator.pth')

    print("Models saved.")


if __name__ == "__main__":
    main()