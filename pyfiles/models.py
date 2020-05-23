import torch
import torchvision
import numpy as np

class Hippocampal():
    """
    HC: Save current task's data  
    """
    def __init__(self, task_num, eps):
        self.images = []
        self.labels = []

        self.task_num = task_num
        self.eps = eps


    def distillate(self, dataloader):
        assert dataloader is torch.utils.data.dataloader.DataLoader
        for data in dataloader:
            image, label = data
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            self.images.append(image)
            self.labels.append(label)

        self.images = torch.cat(self.images)
        self.labels = torch.cat(self.labels)

        if torch.cuda.is_available():
            self.images = self.images.cuda()
            self.labels = self.labels.cuda()


    def forward(self, x):
        beta = torch.zeros((100))
        if torch.cuda.is_available():
            beta = beta.cuda()

        for i, image in enumerate(self.images):
            label = self.labels[i]
            norm = self.eps + ((x - image)**2).sum().sqrt().item()
            if beta[label] > norm:
                beta[label] = norm

        beta = 1 / beta
        beta = beta / beta.sum()

        return beta 


# autoencoder
class VAE(torch.nn.Module):
    def __init__(self, input_data_shape = (28, 28),  hidden_layer_num = 400, latent_dim = 20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_layer_num = input_data_shape[0] * input_data_shape[1]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_layer_num, hidden_layer_num),
            torch.nn.ReLU()
        )

        self.z_mean_out = torch.nn.Linear(hidden_layer_num, latent_dim)
        self.log_var_out = torch.nn.Linear(hidden_layer_num, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_layer_num),
            torch.nn.ReLU(),
            
            torch.nn.Linear(hidden_layer_num, self.input_layer_num),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        z_mean, log_var = self.encode(x)
        latent_variable = lib.reparam_trick(z_mean, log_var)
        return self.decode(latent_variable), z_mean, log_var
    
    def encode(self, x):
        _x = x.view(-1, 28*28)
        out = self.encoder(_x)
        z_mean = self.z_mean_out(out)
        log_var = self.log_var_out(out)
        return z_mean, log_var

    def decode(self, x):
        return self.decoder(x)
            
    def reparam_trick(z_mean, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return z_mean + std * eps

    def vae_loss(x, x_hat, z_mean, log_var):
        bceloss = torch.nn.functional.binary_cross_entropy(x_hat, x.view(x.shape[0], -1), reduction='sum')
        kldloss = (1 + log_var - z_mean**2 - torch.exp(log_var)).sum() / 2
        return bceloss - kldloss


class BLA(torch.nn.Module):
    def __init__(self, input_data_shape):
        super(BLA, self).__init__()
        num_channels, width, height = input_data_shape
        conv2d_1 = torch.nn.Conv2d(in_channels=num_channels, 
                                   out_channels=width*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_2 = torch.nn.Conv2d(in_channels=width*4, 
                                   out_channels=width*8, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.Conv2d(in_channels=width*8, 
                                   out_channels=1, 
                                   kernel_size=7, 
                                   stride=1,
                                   padding=0,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features=width*4),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features=width*8),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_3,
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).view(-1, 1)