#Indlæser mapper
import os
import glob
import argparse
import numpy as np
import re #Ingen ide om hvad det her er? JEG VED GODT HVAD DET ER. re bruger jeg til at finde As i stien, 
import h5py
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from scipy.interpolate import interp1d
from numba import jit
from classy import Class
import matplotlib.pyplot as plt
#Konfiguration-----------------------------------------------------------------
root_dir       = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Varians Test\FejlerDet2"
#output_dir = os.path.join(root_dir, 'all_deltaAS')
ngrid   = 32
boxsize = 500


#Input og output mapper--------------------------

test_input_dir      = os.path.join(root_dir, 'Test_data_CONCEPT')
trainval_input_dir  = os.path.join(root_dir, 'TrainingVal_Concept')

test_output_dir     = os.path.join(root_dir, 'Test data_CONCEPT')
trainval_output_dir = os.path.join(root_dir, 'Training & val data_CONCEPT')



def find_hdf5_files(input_dir):
    """
    Returnerer en sorteret liste af alle filer i input_dir (og undermapper),
    hvis navn ender på .hdf5 eller .h5 (case-insensitive).
    """
    files = []
    for root, _dirs, fnames in os.walk(input_dir):
        for fname in fnames:
            if fname.lower().endswith(('.hdf5','.h5')):
                files.append(os.path.join(root, fname))
    return sorted(files)

#Hapser det vi skal bruge fra HDF5-filerne-----------------------------------
def extract_As_from_path(filepath):
    """
    Go three levels up from the .txt file to NN_As..., then regex out the number.
    """
    for part in filepath.split(os.sep):
        m = re.search(r'As([0-9.+eE-]+)', part)
        if m:
            return m.group(1)
    raise ValueError(f"Kunne ikke finde A_s i stien: {filepath}")


def read_positions(filename):
    """
    Læser alle positionerne fra HDF5-filen. 
    """
    with h5py.File(filename, 'r') as f:
        return f['components/matter/pos'][:]  # Nx3 array
    

def read_boxsize(filename):
    """
    Læser boxsize en gang.
    """
    with h5py.File(filename, 'r') as f:
        return f.attrs['boxsize']
    

#Main body--------------------------------------------------
#Window funktion til vores CIC pipeline 
def cic_window_ft(ngrid, boxsize):
    d   = boxsize/ngrid
    k_vals= 2*np.pi*np.fft.fftfreq(ngrid, d=d)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_quist = np.pi*(d)**(-1)
    Wx = np.sinc(kx/(2*k_quist))
    Wy = np.sinc(ky/(2*k_quist))
    Wz = np.sinc(kz/(2*k_quist))
    W = (Wx*Wy*Wz)**2
    W[W == 0] = 1.0
    #print(W) # Sætter W[0,0,0] til 1 for at undgå division med nul
    return W

#Applier dekonvolutionsmetoden til vores CIC grid
def deconvolve_cic(delta_grid, boxsize, ngrid):
    delta_k = np.fft.fftn(delta_grid)  # Normaliseret til k-rum
    W  = cic_window_ft(ngrid, boxsize)
    delta_k /= W
    return np.fft.ifftn(delta_k).real # Normaliseret til k-rum

@jit(forceobj=True)
def cic_density(positions, boxsize, N):
    # Finder den nærmeste gittercelle for hvert placeret partikel
    delta_cic = np.zeros((N, N, N), dtype=np.float64)  # Gitter til at gemme densiteten
    cell_size = boxsize / N  # Størrelsen af hver gittercelle

    for p in positions:
        i = int(p[0] / cell_size) % N
        j = int(p[1] / cell_size) % N
        k = int(p[2] / cell_size) % N


        # Finder den normaliserede vægt (afstanden til gitterpunktet i hver retning)
        dx = (p[0]%cell_size) / cell_size
        dy = (p[1]%cell_size) / cell_size
        dz = (p[2]%cell_size) / cell_size


        # Fordeler massen over de 8 nærmeste celler
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    weight = ((1 - dx) if di == 0 else dx) * \
                            ((1 - dy) if dj == 0 else dy) * \
                            ((1 - dz) if dk == 0 else dz)

                
                    ni, nj, nk = (i + di) % N, (j + dj) % N, (k + dk) % N  
                    delta_cic[ni, nj, nk] += weight  

    return delta_cic


def compute_delta(density):
    return density/density.mean() - 1.0 

def process_folder_by_id(input_dir, output_dir, prefix, ngrid):
    # 1) Hent alle undermapper og sorter
    all_dirs = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )

    id_dirs = []
    for d in all_dirs:
        full_dir = os.path.join(input_dir, d)
        h5_files = find_hdf5_files(full_dir)
        selected = [f for f in h5_files if 'snapshot_a=1.00' in os.path.basename(f)]

        if selected:
            id_dirs.append((d, selected[0]))  # Gem både mappe og valgt fil
        else:
            print(f"Ingen snapshot i {full_dir}, springer over.")

    print(f" Fundet {len(id_dirs)} mapper med snapshot i {input_dir}")
    if not id_dirs:
        print("Ingen gyldige mapper at behandle. Afbryder.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2) Brug første gyldige fil til at læse boxsize
    sample_file = id_dirs[0][1]
    boxsize = read_boxsize(sample_file)
    print(f" boxsize = {boxsize}\n")

    # 3) Loop over mapper og behandle de valgte filer
    for idx, (d, h5file) in enumerate(id_dirs, start=1):
        print(f"[{idx}/{len(id_dirs)}] Behandler {h5file}")
        pos   = read_positions(h5file)
        dens  = cic_density(pos, boxsize, ngrid)
        delta = compute_delta(dens).astype(np.float64)
        delta = deconvolve_cic(delta, boxsize, ngrid).astype(np.float64)
        print(delta.mean())

        delta = delta[..., None]  # shape (ngrid,ngrid,ngrid,1)

        out_name = f"{prefix}{idx}.npy"
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, delta)
        print(f" Gemte {out_path}")





if __name__ == '__main__':
    # Test data 
    process_folder_by_id(
        test_input_dir,
        test_output_dir,
        'delta_test_id-',
        ngrid
    )

    # Training + Val data 
    process_folder_by_id(
        trainval_input_dir,
        trainval_output_dir,
        'delta_train_id-',
        ngrid
    )





d = np.load(r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Varians test\FejlerDet2\Training & val data_CONCEPT\delta_train_id-1.npy")
print("d=", d.shape,"mean =", d.mean(),"var =", d.var())


p = np.load(r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Varians test\FejlerDet2\Training & val data_CONCEPT\delta_train_id-19.npy")
print("p=", p.shape,"mean =", p.mean(),"var =", p.var())

a = np.load(r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Varians test\FejlerDet2\Test data_CONCEPT\delta_test_id-1.npy")
print("a=", a.shape,"mean =", a.mean(),"var =", a.var())


s = np.load(r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Varians test\FejlerDet2\Test data_CONCEPT\delta_test_id-8.npy")
print("s=", s.shape,"mean =", s.mean(),"var =", s.var())
