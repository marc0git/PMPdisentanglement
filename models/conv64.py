
import torch.nn as nn


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return x

class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x
class SimpleConv64D(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size,with_bn=0):
        super().__init__()

        self._latent_dim = latent_dim
        self._num_channels = num_channels
        self._image_size = image_size
        assert image_size == 64, 'This model only works with image size 64x64.'


        if with_bn:
            self.main = nn.Sequential(
                Unsqueeze3D(),
                nn.Conv2d(latent_dim, 512, 1, 2),
                nn.ReLU(True),
                nn.BatchNorm2d(512),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(256, 128, 4, 2),
                nn.ReLU(True),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 128, 4, 2),
                nn.ReLU(True),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 64, 4, 2),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 64, 4, 2),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, num_channels, 3, 1),
            )
        else:
            self.main = nn.Sequential(
                nn.Linear(latent_dim, 256, bias=True),
                nn.ReLU(True),
                nn.Linear(256, 256, bias=True),
                nn.ReLU(True),
                nn.Linear(256,64*4*4, bias=True),
                nn.ReLU(True),
                nn.Unflatten(1,(64,4,4)),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, num_channels, 4, 2, 1)
            )

        # output shape = bs x 3 x 64 x 64



    def forward(self, x):
        return self.main(x)




class SimpleConv64(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size,batch_size=16,with_bn=0):
        super().__init__()

        self._latent_dim = latent_dim
        self._num_channels = num_channels
        self._image_size = image_size
        self.batch_size=batch_size
        assert image_size == 64, 'This model only works with image size 64x64.'
        
        if with_bn:
            self.main = nn.Sequential(
                nn.Conv2d(num_channels, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(512),
                Flatten3D(),
                nn.Linear(512, latent_dim, bias=True),
            )

        else:#256*256*4*4 + 128*256*4*4 + 64*128*4*4 + 32*64*4*4 + 32*32*4*4 + 32*4*4 + 256*10
            self.main = nn.Sequential(
                nn.Conv2d(num_channels, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
#                 nn.Conv2d(128, 256, 4, 2, 1),
#                 nn.ReLU(True),
#                 nn.Conv2d(256, 256, 4, 2, 1),
#                 nn.ReLU(True),
                Flatten3D(),
                nn.Linear(64*4*4, 256, bias=True),
                nn.ReLU(True),
                nn.Linear(256, latent_dim, bias=True),
            )

    def latent_dim(self):
        return self._latent_dim

    def num_channels(self):
        return self._num_channels

    def image_size(self):
        return self._image_size

    def forward(self, x):
        if len(x.shape)==3:
            x = x[None,...]
        assert(x.shape[-3] == self._num_channels)
        
        return self.main(x)