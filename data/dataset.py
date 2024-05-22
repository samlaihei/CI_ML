import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ImgDataset(Dataset):
    def __init__(self, filelist, transform=None, antenna:int=7):
        self.filelist = filelist
        self.transform = transform
        self.imgs = [np.load(f) for f in self.filelist]
        # self.imgs = np.concatenate(self.imgs)

        self.class_label_names = [f.split('_')[-1].split('.')[0] for f in self.filelist]

        self.class_labels = [np.zeros((len(i), len(self.class_label_names))) for i in self.imgs]

        for i in range(len(self.imgs)):
            self.class_labels[i][:, self.class_label_names.index(self.class_label_names[i])] = 1

        self.imgs = np.concatenate(self.imgs)
        self.class_labels = np.concatenate(self.class_labels)

        # normalise every image
        for i in range(len(self.imgs)):
            # replace NaNs
            self.imgs[i] = np.nan_to_num(self.imgs[i])
            self.imgs[i] = (self.imgs[i] - np.nanmin(self.imgs[i])) / (np.nanmax(self.imgs[i]) - np.nanmin(self.imgs[i]))


        self.uvf = np.load('ehtuv.npz')
        self.antenna = antenna
        self.atriads, self.btriads = self.Triads(self.antenna)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.imgs[idx]
        if self.transform:
            image = self.transform(image)

        class_label = self.class_labels[idx]
        ci = self.FTCI(np.array([image])).reshape(-1)

        image = image.to(dtype=torch.float32)
        class_label = torch.from_numpy(class_label).float()
        ci = torch.from_numpy(ci).float()

        return image, ci, class_label


    def FTCI(self, imgs):
        vis = self.Visibilities(imgs)
        ci = self.ClosureInvariants(vis)
        return ci
    

    def Visibilities(self, imgs:np.ndarray):
        """
        Samples the visibility plane DFT according to eht uv co-ordinates.

        Args:
            imgs (np.ndarray): array of images

        Returns:
            vis (np.ndarray): visibilities taken for each image
        """
        uv = np.concatenate([self.uvf[x] for x in self.uvf])
        vis = self.DFT(imgs, uv)
        return vis.reshape((len(imgs), len(self.uvf), -1))


    def ClosureInvariants(self, vis:np.ndarray, n:int=7):
        """
        Calculates copolar closure invariants for visibilities assuming an n element 
        interferometer array using method 1.

        Nithyanandan, T., Rajaram, N., Joseph, S. 2022 “Invariants in copolar 
        interferometry: An Abelian gauge theory”, PHYS. REV. D 105, 043019. 
        https://doi.org/10.1103/PhysRevD.105.043019 

        Args:
            vis (np.ndarray): visibility data sampled by the interferometer array
            n (int): number of antenna as part of the interferometer array

        Returns:
            ci (np.ndarray): closure invariants
        """

        C_oa = vis[:, :, self.btriads[:, 0]]
        C_ab = vis[:, :, self.btriads[:, 1]]
        C_bo = np.conjugate(vis[:, :, self.btriads[:, 2]])
        A_oab = C_oa / np.conjugate(C_ab) * C_bo
        A_oab = np.dstack((A_oab.real, A_oab.imag))
        A_max = np.nanmax(np.abs(A_oab), axis=-1, keepdims=True)
        ci = A_oab / A_max
        return ci


    def DFT(self, data, uv, xfov=225, yfov=225):
        if data.ndim == 2:
            data = data[None,...]
            out_shape = (uv.shape[0],)
        elif data.ndim > 2:
            data = data.reshape((-1,) + data.shape[-2:])
            out_shape = data.shape[:-2] + (uv.shape[0],)
        ny, nx = data.shape[-2:]
        dx = xfov*4.84813681109536e-12 / nx
        dy = yfov*4.84813681109536e-12 / ny
        angx = (np.arange(nx) - nx//2) * dx
        angy = (np.arange(ny) - ny//2) * dy
        lvect = np.sin(angx)
        mvect = np.sin(angy)
        l, m = np.meshgrid(lvect, mvect)
        lm = np.concatenate([l.reshape(1,-1), m.reshape(1,-1)], axis=0)
        imgvect = data.reshape((data.shape[0],-1))
        x = -2*np.pi*np.dot(uv,lm)[None, ...]
        visr = np.sum(imgvect[:, None, :] * np.cos(x, dtype=np.float32), axis=-1)
        visi = np.sum(imgvect[:, None, :] * np.sin(x, dtype=np.float32), axis=-1)
        if data.ndim == 2:
            vis = visr.ravel() + 1j*visi.ravel()
        else:
            vis = visr.ravel() + 1j*visi.ravel()
            vis = vis.reshape(out_shape)
        return vis

    def Triads(self, n:int):
        """
        Generates arrays of antenna and baseline indicies that form triangular 
        loops pivoted around the 0th antenna. Used to calculate closure invariants
        whereby specific baseline correlations need to be indexed according 
        to those triangular loops.
        Baseline array format [ant1, ant2]:
        [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6] ... 
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6] ...
        [2, 3], [2, 4], [2, 5], [2, 6] ...
        [3, 4], [3, 5], [3, 6] ...
        [4, 5], [4, 6] ...
        [5, 6] ...

        Args:
            n (int): number of antenna in the array

        Returns:
            atriads (np.ndarray): antenna triangular loop indicies
            btriads (np.ndarray): baseline triangular loop indicies
        """
        ntriads = (n-1)*(n-2)//2
        ant1 = np.zeros(ntriads, dtype=np.uint8)
        ant2 = np.arange(1, n, dtype=np.uint8).reshape(n-1, 1) + np.zeros(n-2, dtype=np.uint8).reshape(1, n-2)
        ant3 = np.arange(2, n, dtype=np.uint8).reshape(1, n-2) + np.zeros(n-1, dtype=np.uint8).reshape(n-1, 1)
        anti = np.where(ant3 > ant2)
        ant2, ant3 = ant2[anti], ant3[anti]
        atriads = np.concatenate([ant1.reshape(-1, 1), ant2.reshape(-1, 1), ant3.reshape(-1, 1)], axis=-1)
        
        ant_pairs_01 = list(zip(ant1, ant2))
        ant_pairs_12 = list(zip(ant2, ant3))
        ant_pairs_20 = list(zip(ant3, ant1))
        
        t1 = np.arange(n, dtype=int).reshape(n, 1) + np.zeros(n, dtype=int).reshape(1, n)
        t2 = np.arange(n, dtype=int).reshape(1, n) + np.zeros(n, dtype=int).reshape(n, 1)
        bli = np.where(t2 > t1)
        t1, t2 = t1[bli], t2[bli]
        bl_pairs = list(zip(t1, t2))

        bl_01 = np.asarray([bl_pairs.index(apair) for apair in ant_pairs_01])
        bl_12 = np.asarray([bl_pairs.index(apair) for apair in ant_pairs_12])
        bl_20 = np.asarray([bl_pairs.index(tuple(reversed(apair))) for apair in ant_pairs_20])
        btriads = np.concatenate([bl_01.reshape(-1, 1), bl_12.reshape(-1, 1), bl_20.reshape(-1, 1)], axis=-1)
        return atriads, btriads

