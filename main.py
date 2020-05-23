import argparse
import torch
import torchvision
import numpy as np

# Custom library
import pyfiles.datasets as datasets
import pyfiles.models as models
import pyfiles.lib as lib

parser = argparse.ArgumentParser()
parser.add_argument("--img_type", type=str, default="MNIST", help="Type of image to train")
opt = parser.parse_args()

batch_size = 64

# standardization
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

TrainDataLoaders = []
TestDataLoaders = []

for i in range(10):
    TrainDataSet = datasets.CIFAR100_IncrementalDataset(root='./data',
                                                        train=True,
                                                        transform=transform,
                                                        download=True,
                                                        classes=range(i * 10, (i+1) * 10))
                
    TestDataSet = datasets.CIFAR100_IncrementalDataset(root='./data', 
                                                        train=False,
                                                        transform=transform,
                                                        download=True,
                                                        classes=range(i * 10, (i+1) * 10))
                
    TrainDataLoaders.append(torch.utils.data.DataLoader(TrainDataSet,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=2))
    
    TestDataLoaders.append(torch.utils.data.DataLoader(TestDataSet, 
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=2))


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#base_knowledge = torchvision.models.resnet152()
epochs = 200

hc = models.Hippocampal()
mPFC = models.VAE()
BLA = models.BLA()
optim = torch.optim.Adam(mPFC.parameters())

if torch.cuda.is_available():
    base_knowledge = base_knowledge.cuda()
    crit = crit.cuda()
    mPFC = mPFC.cuda()
    BLA = BLA.cuda()

for t in range(5):
    TrainDataLoader = TrainDataLoaders[t]
    for (x, y) in TrainDataLoader:
        x = x.to(device)
        y = y.to(device)
        ### mPFC Training

        ### BLA Training
        A = BLA(x)
        
        phi = 1/(1-A)
        pass