import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

from base.base_net import BaseNet

class CyCIF_ResidualNetwork(BaseNet):
    def __init__(self):
        super(CyCIF_ResidualNetwork, self).__init__()

        self.rep_dim = 512 # latent dimension
 
        # Load pretrained ResNet model - conv layers has bias = False by default
        resnet50 = models.resnet50(pretrained=True)
        number_features = resnet50.fc.in_features

        # Build Encoder
        modules = list(resnet50.children())[:-1]
        resnet50 = nn.Sequential(*modules)

        for param in resnet50.parameters():
            param.requires_grad = False
        
        self.encoder = resnet50
        self.encoder_output = nn.Linear(number_features*4, self.rep_dim, bias=False)

    def forward(self, x):
        # Encoder (same as the Deep SVDD network above)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  
        x = self.encoder_output(x)

        return x
        
class CyCIF_ResNetAE(BaseNet):

    def __init__(self):
        super(CyCIF_ResNetAE, self).__init__()

        self.rep_dim = 512 # latent dimension
        hidden_dims = [32, 64, 128, 256, 512]
 
        # Load pretrained ResNet model - conv layers has bias = False by default
        resnet50 = models.resnet50(pretrained=True)
        number_features = resnet50.fc.in_features

        # Build Encoder
        modules = list(resnet50.children())[:-1]
        resnet50 = nn.Sequential(*modules)

        for param in resnet50.parameters():
            param.requires_grad = False
        
        self.encoder = resnet50
        self.encoder_output = nn.Linear(number_features*4, self.rep_dim, bias=False)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.rep_dim, hidden_dims[-1]*64, bias=False)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3, 
                                       stride = 2,
                                       padding=1,
                                       output_padding=1,
                                       bias=False),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[i + 1], hidden_dims[i + 1],
                              kernel_size= 3,
                              stride= 1,
                              padding  = 1,
                              bias=False),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1,
                                               bias=False),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], 
                                    out_channels= 3,
                                    kernel_size= 3,
                                    padding= 1,
                                    bias=False),
                            # nn.Tanh())
                            nn.Sigmoid())
        

    def forward(self, x):
        # Encoder (same as the Deep SVDD network above)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  
        x = self.encoder_output(x)

        # Decoder
        x = self.decoder_input(x)
        x = x.view(-1, 512, 8, 8)
        x = self.decoder(x)
        x = self.final_layer(x)

        return x
        


