import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class CyCIF_VanillaNetwork(BaseNet):

    def __init__(self):
        super(CyCIF_VanillaNetwork, self).__init__()

        self.rep_dim = 512 # latent dimension
        in_channels = 1

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                            kernel_size= 3, 
                            stride= 2,
                            padding  = 1,
                            bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Conv2d(h_dim, out_channels=h_dim,
                            kernel_size= 3,
                            stride= 1,
                            padding  = 1,
                            bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules) 
        self.encoder_output = nn.Linear(hidden_dims[-1]*64, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  
        x = self.encoder_output(x)

        return x

class CyCIF_VanillaAE(BaseNet):

    def __init__(self):
        super(CyCIF_VanillaAE, self).__init__()

        self.rep_dim = 512 # latent dimension
        in_channels = 1

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]


        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, 
                              stride= 2,
                              padding  = 1,
                              bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Conv2d(h_dim, out_channels=h_dim,
                              kernel_size= 3,
                              stride= 1,
                              padding  = 1,
                              bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules) 
        self.encoder_output = nn.Linear(hidden_dims[-1]*64, self.rep_dim, bias=False) # CHANGE FOR DIFFERENT DATASETS

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.rep_dim, hidden_dims[-1]*64, bias=False) # CHANGE FOR DIFFERENT DATASETS

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
                                    out_channels= 1,
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