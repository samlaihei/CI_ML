from torch import nn
import torch
import numpy as np
# from VQVAE.vqvae import VQVAE
from VQVAE.encoder import Encoder
from VQVAE.decoder import Decoder
from VQVAE.quantizer import VectorQuantizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vae_FD(nn.Module):
    def __init__(self, latent_size:int=4, k_classes:int=5, imgdim1:int=64, imgdim2:int=64, antenna:int=7):
        super(Vae_FD, self).__init__()

        self.latent_size = latent_size
        self.k_classes = k_classes
        self.imgdim1, self.imgdim2 = imgdim1, imgdim2
        self.imgdim = int(imgdim1*imgdim2)
        self.antenna = antenna

        # Encoder VQVAE # input img, output n latent features
        self.encoder = Encoder(1, self.latent_size, n_res_layers=2, res_h_dim=8)

        self.pre_quantization_conv = nn.Conv2d(
            self.latent_size, self.latent_size//2, kernel_size=1, stride=1)
        
        self.vector_quantization = VectorQuantizer(
            n_e=128, e_dim=self.latent_size//2, beta=0.25)

        # Decoder VQVAE # input n, output img
        self.decoder = Decoder(self.latent_size//2, self.latent_size, n_res_layers=2, res_h_dim=8)

        # take ci and get latent features
        self.mlp = nn.Sequential( # input is size of FTCI, which is (antenna-1)(antenna-2)*4 in this case, output is n
            nn.Linear(int((self.antenna-1)*(self.antenna-2)*4), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.latent_size//2*16**2),
        )

        self.ci_latent = nn.Sequential(
            nn.ConvTranspose2d(int((self.antenna-1)*(self.antenna-2)*4), 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.latent_size//2, 4, 2, 1),
        )

        # linear classifier
        self.classifier = nn.Sequential( # input is n, output is k one-hot classes
            nn.Linear(self.latent_size//2*16**2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.k_classes),
            nn.Softmax(dim=1)
        )    
    
    def forward(self, imgs, ci): 
        features_vae, features_q, recon_img = self.encoder_to_img(imgs)

        features_ci = self.ci_latent(ci.unsqueeze(2).unsqueeze(3))

        pred_img = self.predict(ci)

        features_cls = features_q.view(-1, features_q.shape[1]*features_q.shape[2]*features_q.shape[3])
        pred_class = self.classifier(features_cls)

        return features_vae, features_q, features_ci, recon_img, pred_img, pred_class
    
    def predict(self, ci): # get reconstructed image from ci
        features_pred = self.ci_latent(ci.unsqueeze(2).unsqueeze(3))
        # features_pred = self.pre_quantization_conv(features_pred)
        # _, features_pred, _, _, _ = self.vector_quantization(features_pred)
        recon_img = self.decoder(features_pred)
        return recon_img
    
    def predict_class(self, ci):
        features_cls = self.ci_latent(ci.unsqueeze(2).unsqueeze(3))
        # features_cls = self.pre_quantization_conv(features_cls)
        # _, features_cls, _, _, _ = self.vector_quantization(features_cls)
        features_cls = features_cls.view(-1, features_cls.shape[1]*features_cls.shape[2]*features_cls.shape[3])
        pred_class = self.classifier(features_cls)
        return pred_class
    
    def encoder_to_img(self, imgs):
        features_vae = self.encoder(imgs)
        features_q = self.pre_quantization_conv(features_vae)
        #_, features_q, _, _, _ = self.vector_quantization(features_q)
        recon_img = self.decoder(features_q)
        return features_vae, features_q, recon_img


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
    vae = Vae_FD(latent_size=latent_size, k_classes=5)
    # features_vae, features_ci, recon_img, pred_class = vae(x, ci)
    features_vae, features_q, features_ci, recon_img, pred_img, pred_class = vae(x, ci)

    print('Features vae shape:', features_vae.shape)
    print('Features q shape:', features_q.shape)
    print('Features ci shape:', features_ci.shape)
    print('Recon img shape:', recon_img.shape)
    print('Pred img shape:', pred_img.shape)
    print('Pred class shape:', pred_class.shape)
