from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from VQVAE.encoder import Encoder
from VQVAE.decoder import Decoder
from PyTorch_VAE.vanilla_vae import VanillaVAE



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VaeVanilla(nn.Module):
    def __init__(self, latent_size:int=512, k_classes:int=5, imgdim:int=64, antenna:int=7, hidden_dims:list=None):
        super(VaeVanilla, self).__init__()

        self.latent_size = latent_size
        self.k_classes = k_classes
        self.imgdim = imgdim
        self.antenna = antenna

        self.vae = VanillaVAE(in_channels = 1, latent_dim=latent_size, hidden_dims=hidden_dims)

        # take ci and get latent features
        self.ci_latent = nn.Sequential(
            nn.Linear(int((antenna-1)*(antenna-2)*4), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_size*2)
        )


        # linear classifier
        self.classifier = nn.Sequential( # input is n, output is k one-hot classes
            nn.Linear(self.latent_size*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.k_classes),
            nn.Softmax(dim=1)
        )    

    def forward(self, imgs, ci):
        recon_img, _, mu, log_var = self.vae(imgs)
        features_vae = torch.cat((mu, log_var), dim=1)
        features_ci, pred_img = self.predict(ci)
        pred_class = self.predict_class(ci)
        return features_vae, features_ci, recon_img, pred_img, pred_class
    
    
    def predict(self, ci): # get reconstructed image from ci
        features_pred = self.ci_latent(ci)
        mu, log_var = torch.chunk(features_pred, 2, dim=1)
        z = self.vae.reparameterize(mu, log_var)
        pred_img = self.vae.decode(z)
        return features_pred, pred_img
    
    def predict_class(self, ci):
        features_cls = self.ci_latent(ci)
        pred_class = self.classifier(features_cls)
        return pred_class


if __name__ == "__main__":
    N = 10
    imgdim = 64
    latent_size = 512

    # random data
    x = np.random.random_sample((N, 1, imgdim, imgdim))
    x = torch.tensor(x).float()#.to(device)

    ci = np.random.random_sample((N, 120))
    ci = torch.tensor(ci).float()#.to(device)

    # test vae
    vae = VaeVanilla(latent_size=latent_size)
    features_vae, features_ci, recon_img, pred_img, pred_class = vae(x, ci)

    print("features_vae", features_vae.shape)
    print("features_ci", features_ci.shape)
    print("recon_img", recon_img.shape)
    print("pred_img", pred_img.shape)
    print("pred_class", pred_class.shape)

