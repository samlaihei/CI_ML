from torch import nn
import torch
import numpy as np
from VQVAE.encoder import Encoder
from VQVAE.decoder import Decoder
from vector_quantize_pytorch import VectorQuantize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vae_FD(nn.Module):
    def __init__(self, latent_size:int=4, k_classes:int=5, imgdim1:int=64, imgdim2:int=64, antenna:int=7):
        super(Vae_FD, self).__init__()

        self.latent_size = latent_size
        self.k_classes = k_classes
        self.imgdim1, self.imgdim2 = imgdim1, imgdim2
        self.imgdim = int(imgdim1*imgdim2)
        self.antenna = antenna
        self.ci_dim = int((self.antenna-1)*(self.antenna-2)*4)

        # Encoder VQVAE # input img, output n latent features
        self.encoder = Encoder(1, self.latent_size, n_res_layers=2, res_h_dim=8)

        self.pre_quantization_conv_E = nn.Conv2d(
            self.latent_size, self.latent_size//2, kernel_size=1, stride=1)
        
        self.pre_quantization_conv_CI = nn.Conv2d(
            self.latent_size, self.latent_size//2, kernel_size=1, stride=1)
        
        self.vector_quantization_E = VectorQuantize(
            dim = 2,
            codebook_size = 256,
            accept_image_fmap = True,                   # set this true to be able to pass in an image feature map
            orthogonal_reg_weight = 10,                 # in paper, they recommended a value of 10
            orthogonal_reg_max_codes = 128,             # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
            orthogonal_reg_active_codes_only = False    # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
        )
        

        # Decoder VQVAE # input n, output img
        self.decoder = Decoder(self.latent_size, self.latent_size, n_res_layers=2, res_h_dim=8)

        # take ci and get latent features
        self.ci_latent_FD = nn.Sequential(
            nn.Conv2d(self.ci_dim,128,kernel_size=1,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(8,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

        self.ci_mlp = nn.Sequential(
            nn.Linear(self.ci_dim, 256),
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
            nn.Linear(2048, self.imgdim),
            nn.ReLU(),
        )


        # linear classifier
        self.classifier = nn.Sequential( # input is n, output is k one-hot classes
            nn.Linear(self.latent_size*16**2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.k_classes),
            nn.Softmax(dim=1)
        )   

    def ci_latent(self, ci):
        # features_ci = self.ci_latent_FD(ci)
        ci = ci.reshape(-1, self.ci_dim)
        features_ci = self.ci_mlp(ci)
        features_ci = features_ci.reshape(-1, 1, self.imgdim1, self.imgdim2)
        features_ci = self.encoder(features_ci)
        return features_ci
    
    def forward(self, imgs, ci): 
        features_vae, features_q, recon_img = self.encoder_to_img(imgs)
        features_ci, pred_img = self.predict(ci)

        features_cls = features_q.reshape(-1, features_q.shape[1]*features_q.shape[2]*features_q.shape[3])

        pred_class = self.classifier(features_cls)

        return features_vae, features_q, features_ci, recon_img, pred_img, pred_class
    
    def predict(self, ci): # get reconstructed image from ci
        features_pred = self.ci_latent(ci.reshape(-1, int((self.antenna-1)*(self.antenna-2)*4), 1, 1))
        # features_pred = self.pre_quantization_conv_E(features_pred)
        # features_pred, _, _ = self.vector_quantization_E(features_pred)
        pred_img = self.decoder(features_pred)
        # pred_img = self.mlp(ci).reshape(-1, 1, self.imgdim1, self.imgdim2)
        return features_pred, pred_img
    
    def predict_class(self, ci):
        features_cls = self.ci_latent(ci.reshape(-1, int((self.antenna-1)*(self.antenna-2)*4), 1, 1))
        # features_cls = self.pre_quantization_conv_E(features_cls)
        # features_cls, _, _ = self.vector_quantization_E(features_cls)
        features_cls = features_cls.reshape(-1, features_cls.shape[1]*features_cls.shape[2]*features_cls.shape[3])
        pred_class = self.classifier(features_cls)
        return pred_class
    
    def encoder_to_img(self, imgs):
        features_vae = self.encoder(imgs)
        features_q = features_vae
        # features_q = self.pre_quantization_conv_E(features_vae)
        # features_q, _, _ = self.vector_quantization_E(features_q)
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
    ci[0, :] = 0

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


