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

class VGG_VAE(nn.Module):
    def __init__(self, input_size=60, output_size=64):
        super(VGG_VAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=14, stride=2, padding=1),
            nn.Tanh()            
        )

        self.fc_out = nn.Sequential(
            #nn.Linear(self.input_size, 3 * output_size * output_size),
            nn.Linear(self.input_size, 512 * 4 * 4),
            nn.Tanh()
        )
    def forward(self, x, noise):
        #print(x.size())
        fc = self.fc_out(x)

        fc = fc.view(fc.size(0), 512, 4, 4)
        noise = noise + fc
        decoded = self.decoder(noise)

        #print(decoded.size())
        output = decoded
        #fc = fc.view(fc.size(0), 3, self.output_size, self.output_size)
        #output = fc + decoded

        return output