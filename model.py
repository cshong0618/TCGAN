import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, output_class=5):
        super(Discriminator, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 1),
            nn.Conv2d(24, 24, 5, 2, 1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Dropout2d(),
            nn.Conv2d(24, 48, 5, 1, 1),
            nn.Conv2d(48, 48, 5, 1, 1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Dropout2d()
        )

        self.classifier = nn.Sequential(
            nn.Linear(48 * 5 * 5, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_class),
        )

    def forward(self, X):
        x = self.feature(X)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #print(x.size())

        return x

class Generator(nn.Module):
    def __init__(self, input_class=5):
        super(Generator, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 48, 8, 2, 1),
            nn.LeakyReLU(),            
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(),            
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(),    
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(),            
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(),            
            nn.ConvTranspose2d(48, 48, 6, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(),            
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(),
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(),            
            nn.Conv2d(48, 48, 1),
            nn.LeakyReLU(),    
            nn.ConvTranspose2d(48, 24, 6, 2, 1),
            nn.LeakyReLU(),            
            nn.ConvTranspose2d(24, 3, 8, 2, 1),
            nn.Tanh(),
        )

        self.feature_conditioner = nn.Sequential(
            nn.Linear(input_class, 1 * 5 * 5),
            nn.LeakyReLU()
        )

        self.feature_conv = nn.Sequential(
            nn.Conv2d(1, 12, 1),
            nn.LeakyReLU(),
            nn.Conv2d(12, 24, 1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 48, 1),
            nn.LeakyReLU()
        )

    def forward(self, x, noise):
        x = self.feature_conditioner(x)
        #print(x.size())
        x = x.view(x.size(0), 1, 5, 5)
        x = self.feature_conv(x)
        noise = noise + x

        output = self.decoder(noise)

        #print(output.size())

        return output