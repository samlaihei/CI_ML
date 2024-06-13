from torch import nn
import torch
import numpy as np
# from VQVAE.vqvae import VQVAE
# from VQVAE.encoder import Encoder
# from VQVAE.decoder import Decoder
# from VQVAE.quantizer import VectorQuantizer
# from vector_quantize_pytorch import VectorQuantize
# from vector_quantize_pytorch import LatentQuantize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vae_FD(nn.Module):
    def __init__(self, latent_size:int=4, k_classes:int=5, imgdim1:int=64, imgdim2:int=64, antenna:int=7):
        super(Vae_FD, self).__init__()

        self.latent_size = latent_size
        self.k_classes = k_classes
        self.imgdim1, self.imgdim2 = imgdim1, imgdim2
        self.imgdim = int(imgdim1*imgdim2)
        self.antenna = antenna


        # take ci and get latent features
        self.mlp = nn.Sequential( # input is size of FTCI, which is (antenna-1)(antenna-2)*4 in this case, output is n
            nn.Linear(int((self.antenna-1)*(self.antenna-2)*4), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
        )

        # self.ci_latent = nn.Sequential(
        #     nn.ConvTranspose2d(int((self.antenna-1)*(self.antenna-2)*4), 64, 4, 1, 0),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(32, self.latent_size, 4, 2, 1),
        # )


        # # linear classifier
        # self.classifier = nn.Sequential( # input is n, output is k one-hot classes
        #     nn.Linear(self.latent_size//2*16**2, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.k_classes),
        #     nn.Softmax(dim=1)
        # )    
    
    def forward(self, ci): # get reconstructed image from ci
        pred_img = self.mlp(ci).reshape(-1, 1, self.imgdim1, self.imgdim2)
        return pred_img


if __name__ == "__main__":
    N = 10
    imgdim = 64
    latent_size = 4

    # random data
    x = np.random.random_sample((N, 1, imgdim, imgdim))
    x = torch.tensor(x).float()#.to(device)

    ci = np.random.random_sample((N, 120))
    ci = torch.tensor(ci).float()#.to(device)

    # test vae
    mlp = Vae_FD(latent_size=latent_size, k_classes=5)
    # features_vae, features_ci, recon_img, pred_class = vae(x, ci)
    pred_img = mlp(ci)

    print('Pred img shape:', pred_img.shape)


