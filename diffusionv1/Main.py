import torch
import torchvision
from diffusionv1.Trainer import train
from diffusionv1.Unet import Unet
import torchvision.transforms as transforms


def main():
    NO_EPOCHS = 2000
    LR = 0.001
    BATCH_SIZE = 128

    #data and transform #taken from https://github.com/dtransposed/code_videos/blob/main/Diffusion%20Model.ipynb
    transform = transforms.Compose([
        transforms.Resize(32), # Resize the input image
        transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
        transforms.Lambda(lambda t: (t * 2) - 1), # Scale data between [-1, 1] 
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

    model = Unet(dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train(model, trainloader, testloader, optimizer, NO_EPOCHS, BATCH_SIZE)

