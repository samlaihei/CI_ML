import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import data.CI as CI

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ImgDataset(Dataset):
    def __init__(self, filelist, transform=None,
                 ehtim=False, ehtarray='./data/EHT2017.txt', subarray=None,
                 date='2017-04-05', ra=187.7059167, dec=12.3911222, bw_hz=[230e9],
                 tint_sec=10, tadv_sec=48*60, tstart_hr=4.75, tstop_hr=6.5,
                 noise=False, sgrscat=False, ampcal=True, phasecal=True):
        self.filelist = filelist
        self.transform = transform
        self.imgs = [np.load(f) for f in self.filelist]
        self.closure = CI.Closure_Invariants(filename='./data/ehtuv.npz',
                                             ehtim=ehtim, ehtarray=ehtarray, subarray=subarray,
                                             date=date, ra=ra, dec=dec, bw_hz=bw_hz,
                                             tint_sec=tint_sec, tadv_sec=tadv_sec, tstart_hr=tstart_hr, tstop_hr=tstop_hr,
                                             sgrscat=sgrscat, ampcal=ampcal, phasecal=phasecal)
        
        # self.imgs = np.concatenate(self.imgs)

        self.class_label_names = [f.split('_')[-1].split('.')[0] for f in self.filelist]

        self.class_labels = [np.zeros((len(i), len(self.class_label_names))) for i in self.imgs]

        for i in range(len(self.imgs)):
            self.class_labels[i][:, self.class_label_names.index(self.class_label_names[i])] = 1

        self.imgs = np.concatenate(self.imgs)
        self.class_labels = (np.concatenate(self.class_labels))

        # normalise every image
        for i in range(len(self.imgs)):
            # replace NaNs
            self.imgs[i] = np.nan_to_num(self.imgs[i])
            self.imgs[i] = (self.imgs[i] - np.nanmin(self.imgs[i])) / (np.nanmax(self.imgs[i]) - np.nanmin(self.imgs[i]))


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.imgs[idx]
        if self.transform:
            image = self.transform(image)

        class_label = self.class_labels[idx]
        ci = self.closure.FTCI(np.array([image])).reshape(-1)

        image = image.to(dtype=torch.float32)
        class_label = torch.from_numpy(class_label).float()
        ci = torch.from_numpy(ci).float()

        return image, ci, class_label
