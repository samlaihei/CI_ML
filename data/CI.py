import numpy as np
import ehtim as eh
from astropy.time import Time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Closure_Invariants():

    def __init__(self, filename='ehtuv.npz', ehtim=False, ehtarray='EHT2017.txt', subarray=None,
                 date='2017-04-05', ra=187.7059167, dec=12.3911222, bw_hz=[230e9],
                 tint_sec=10, tadv_sec=48*60, tstart_hr=4.75, tstop_hr=6.5,
                 noise=False, sgrscat=False, ampcal=True, phasecal=True):
        
        self.ehtim = ehtim

        if ehtim:
            t = Time(date, format='iso', scale='utc')
            self.mjd = int(t.mjd)
            self.ra = ra/360*24
            self.dec = dec
            self.bw_hz = bw_hz
            self.tint_sec = tint_sec
            self.tadv_sec = tadv_sec
            self.tstart_hr = tstart_hr
            self.tstop_hr = tstop_hr
            self.noise = noise
            self.sgrscat = sgrscat
            self.ampcal = ampcal
            self.phasecal = phasecal

            self.ehtarray = eh.array.load_txt(ehtarray)
            if subarray is not None:
                self.ehtarray = ehtarray.make_subarray(['ALMA','APEX','LMT','PV','SMT','JCMT','SMA'])
            
            template = './data/template_sgra.txt'
            template = eh.image.load_txt(template)
            template.ra = self.ra
            template.dec = self.dec
            template.mjd = self.mjd
            self.template = template.regrid_image(template.fovx(), 64, 'cubic')

            self.obslist = []
            for bw in self.bw_hz:
                template.rf = bw
                obs = self.template.observe(self.ehtarray, self.tint_sec, self.tadv_sec, self.tstart_hr, self.tstop_hr, bw,
                                            mjd = self.mjd, timetype='UTC', ttype='DFT', noise=False, verbose=False)
                self.obslist.append(obs)

            uvlist = []
            antenna_list = []
            for obs in self.obslist:
                for tdata in obs.tlist():
                    num_antenna = len(np.unique(tdata['t1'])) + 1
                    if num_antenna < 3:
                        continue
                    antenna_list.append(num_antenna)
                    u = tdata['u']
                    v = tdata['v']
                    uvlist.append(np.stack((u,v), axis=-1))

            self.uvlist = uvlist
            self.antenna_list = antenna_list
            self.uvf = [0]
        else:
            self.uvf = np.load(filename)    
            self.antenna = 7
            self.atriads, self.btriads = self.Triads(self.antenna)

    def FTCI(self, imgs):
        if self.ehtim:
            ci = np.array([])
            for imgdata in imgs:
                for uv, num_antenna in zip(self.uvlist, self.antenna_list):
                    vis = self.Visibilities(np.array([imgdata]), uv)
                    temp_ci = self.ClosureInvariants(vis, num_antenna)
                    temp_ci = temp_ci.reshape(-1)
                    ci = np.concatenate((ci, temp_ci), axis=0)
            ci = np.array(ci)

        else:
            vis = self.Visibilities(imgs)
            ci = self.ClosureInvariants(vis)
        return ci
    

    def Visibilities(self, imgs:np.ndarray, uv=None):
        """
        Samples the visibility plane DFT according to eht uv co-ordinates.

        Args:
            imgs (np.ndarray): array of images

        Returns:
            vis (np.ndarray): visibilities taken for each image
        """
        if not self.ehtim:
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
        if self.ehtim:
            self.atriads, self.btriads = self.Triads(n)
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

