import torch
import torchvision
import numpy as np

class MNIST_IncrementalDataset(torchvision.datasets.MNIST):
    """
    MNIST Dataset for Incremental Learning
    """
    def __init__(self, 
                 source='./mnist_data', 
                 train=True,
                 transform=None,
                 download=False,
                 classes=range(10)):
        
        super(MNIST_IncrementalDataset, self).__init__(source, 
                                                       train, 
                                                       transform, 
                                                       download=True)
        self.train = train

        if train:
            train_data = []
            train_labels = []
            for i in range(len(self.train_data)):
                if self.train_labels[i] in classes:
                    train_data.append(self.train_data[i].type(dtype=torch.float32))
                    train_labels.append(self.train_labels[i])
            
            self.TrainData = train_data
            self.TrainLabels = train_labels

        else:
            test_data = []
            test_labels = []
            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    test_data.append(self.test_data[i].type(dtype=torch.float32))
                    test_labels.append(self.test_labels[i])
            
            self.TestData = test_data
            self.TestLabels = test_labels

    def __getitem__(self, index):
        if self.train:
            return self.TrainData[index], self.TrainLabels[index]
        else:
            return self.TestData[index], self.TestLabels[index]

    def __len__(self):
        if self.train:
            return len(self.TrainLabels)
        else:
            return len(self.TestLabels)


class CIFAR100_IncrementalDataset(torchvision.datasets.CIFAR100):
    def __init__(self, 
                 root='./cifar100_data', 
                 train=True,
                 transform=None,
                 download=False,
                 classes=range(100)):
        
        super(CIFAR100_IncrementalDataset, self).__init__(source, 
                                                       train, 
                                                       transform, 
                                                       download=True)
        self.train = train

        if train:
            train_data = []
            train_labels = []
            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i].astype(dtype=np.float32))
                    train_labels.append(self.targets[i])
            
            self.TrainData = train_data
            self.TrainLabels = train_labels

        else:
            test_data = []
            test_labels = []
            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i].astype(dtype=np.float32))
                    test_labels.append(self.targets[i])
            
            self.TestData = test_data
            self.TestLabels = test_labels

    def __getitem__(self, index):
        if self.train:
            return self.TrainData[index], self.TrainLabels[index]
        else:
            return self.TestData[index], self.TestLabels[index]

    def __len__(self):
        if self.train:
            return len(self.TrainLabels)
        else:
            return len(self.TestLabels)