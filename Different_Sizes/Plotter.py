from classy import Class
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import scipy.interpolate as scint
import h5py
import glob
import os
import re
from numba import njit
import math
from matplotlib.lines import Line2D
import multiprocessing as mp
# Sætter kassen op
z = 0  # Rødforskydningen, ændres globalt her!
L = 1000 #jeg ved ikke hvorfor den gerne vil være en float, men jeg har fundet at jeg får de bedste resultater når det er en float
N = 64  # Grid size

#Starter på den stokastiske metode (FFT)--------------------------------------------------------------------------------------------------------------------------------
# Funktion til at hente P(k) fra CLASS
def get_matter_power_spectrum(z, k = np.logspace(-4, 0, 500), returnk=False, plot = False):
    params = {
        'output': 'mPk',
        'H0': 67.36,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'Omega_k': 0.0,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'z_pk': z,  # Bruger den globale z-værdi
        'P_k_max_h/Mpc': 10,
        'non linear': 'none',     
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    k_vals = k.flatten()
    Pk = np.array([cosmo.pk(ka, z) for ka in k_vals])


 # Plotter P(k) hvis det står til true 
    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.loglog(k_vals, Pk, lw=2, color='navy')

        plt.xlabel(r'$k\ \left[h/\mathrm{Mpc}\right]$', fontsize=14)
        plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc}/h)^3\right]$', fontsize=14)
        plt.title(f"Linear Matter Power Spectrum at z = {params['z_pk']}", fontsize=16)

        plt.grid(True, which='both', ls='--', alpha=0.7)
        plt.xlim(1e-4, 1e0)
        plt.tight_layout()

    cosmo.struct_cleanup()

    if returnk == True:
        return k_vals, Pk #D_z  
    else:
        return Pk


#Opsætter koordinater 
x_kord = np.linspace(0, L, N, endpoint=False)  #Punkterne i x-retning
y_kord = np.linspace(0, L, N, endpoint=False)  #Punkterne i y-retning
z_kord = np.linspace(0, L, N, endpoint=False)  #Punkterne i z-retning
X, Y, Z = np.meshgrid(x_kord, y_kord, z_kord, indexing ='ij')    # Laver det til et grid

#Opsætter K-vektorer
k_vals = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing ='ij')
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
k_mag[0, 0, 0] = 1e-10  # Undgå division med 0

k_class, Pk_class = get_matter_power_spectrum(z, k = k_mag, returnk=True) 
pk_interp = scint.interp1d(k_class, Pk_class, bounds_error=False, fill_value="extrapolate") #kind = "cubic", måske er der en fejl her.

#Interpolerer P(k) til et grid
k_mag_clipped = np.clip(k_mag, k_class.min(), k_class.max())  # Sikrer ingen out-of-bounds fejl
P_k_grid = pk_interp(k_mag_clipped) #Omdanner til funktion af k-vektor, måske er det her ikke rigtigt. Altså fejlen kan ligge her. 

#Laver et stokastisk komplekst felt
c_1 = np.random.normal(0, 1/np.sqrt(2), size=(N, N, N))
c_2 = np.random.normal(0, 1/np.sqrt(2), size=(N, N, N))
R = c_1 + 1j * c_2  # Det stokastiske komplekse felt R
R[0,0,0] = R[0,0,0].real  # Undgår division med nul
for i in np.arange(N):
    for j in np.arange(N):
        for k in np.arange(N):
            i_sym = (-i)%N
            j_sym = (-j)%N
            k_sym = (-k)%N
            if (i,j,k) != (i_sym,j_sym,k_sym):
                R[i_sym,j_sym,k_sym] = np.conj(R[i,j,k])
#påtvinger nyquist frekvenser at være reele for at opfylde realititets kravet.


#danner delta_k, derefter delta_x og til sidst delta_x i real rum
delta_k = np.sqrt(P_k_grid) 
delta_x = np.fft.ifftn(delta_k * R)  
delta_x = delta_x.real  
#Finder displacement af hver k-vektor i hver retning. 
psi_kx = 1j * kx /k_mag**2 * delta_k 
psi_ky = 1j * ky /k_mag**2 * delta_k 
psi_kz = 1j * kz /k_mag**2 * delta_k 

#Finder displacement i real rum 
psi_x = np.fft.ifftn(psi_kx * R).real
psi_y = np.fft.ifftn(psi_ky * R).real
psi_z = np.fft.ifftn(psi_kz * R).real






#Finder concept's powerspektrum fra snapshots--------------------------------------------------------------------------------------------------------------------------------
def find_all_hdf5(run_dir):
    snaps = []
    for dirpath, _dirs, files in os.walk(run_dir):
        for fn in files:
            if fn.lower().endswith((".h5", ".hdf5")):
                snaps.append(os.path.join(dirpath, fn))
    return sorted(snaps)


@njit
def cic_assign(delta, coords, dx, Nmesh):
    for idx in range(coords.shape[0]):
        x, y, z = coords[idx]
        i = int(x/dx) % Nmesh
        j = int(y/dx) % Nmesh
        k = int(z/dx) % Nmesh
        fx, fy, fz = (x % dx)/dx, (y % dx)/dx, (z % dx)/dx
        for di in (0,1):
            for dj in (0,1):
                for dk in (0,1):
                    w = ((1-fx) if di==0 else fx) * \
                        ((1-fy) if dj==0 else fy) * \
                        ((1-fz) if dk==0 else fz)
                    delta[(i+di)%Nmesh,
                          (j+dj)%Nmesh,
                          (k+dk)%Nmesh] += w
    return delta


def compute_power_spectrum_from_snapshot(snapshot_path, boxsize, Nmesh):
    # 
    with h5py.File(snapshot_path, 'r') as f:
        coords = f['components/matter/pos'][:]

    # 
    k_vals = 2*np.pi * np.fft.fftfreq(Nmesh, d=boxsize/Nmesh)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    # 
    k_quist = np.pi * Nmesh / boxsize
    W_CIC = (np.sinc(kx/(2*k_quist)) *
             np.sinc(ky/(2*k_quist)) *
             np.sinc(kz/(2*k_quist)))**2
    # 
    W_CIC = np.clip(W_CIC, 1e-3, None)
    
    # 
    delta = np.zeros((Nmesh,)*3, dtype=np.float32)
    dx = boxsize / Nmesh

    delta = cic_assign(delta, coords, dx, Nmesh)
    # 
    rho_mean = coords.shape[0] / Nmesh**3
    delta = delta / rho_mean - 1.0

    # 
    delta_k = np.fft.fftn(delta)
    delta_k /= W_CIC

    # 
    V  = boxsize**3
    Pk = np.abs(delta_k)**2 * V / (Nmesh**6)

    # 
    k_min = 2*np.pi / boxsize
    k_max = k_mag.max()
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 300)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    k_flat  = k_mag.ravel()
    Pk_flat = Pk.ravel()
    mask0   = k_flat > 0
    k_flat  = k_flat[mask0]
    Pk_flat = Pk_flat[mask0]

    # 
    counts, _ = np.histogram(k_flat, bins=k_bins)
    Psum,   _ = np.histogram(k_flat, bins=k_bins, weights=Pk_flat)

    mask = counts > 0
    Pk_binned = np.zeros_like(k_centers)
    Pk_binned[mask] = Psum[mask] / counts[mask]

    return k_centers[mask], Pk_binned[mask]


# Hovedskriptet---------------------------------------------------------------------------------------------------------------------------------

root_dir  = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_Sizes\newrun2"
run_dirs  = sorted(glob.glob(os.path.join(root_dir, "3giant*")))


if __name__ == '__main__':
    
    run_dirs = sorted(glob.glob(os.path.join(root_dir, "3giant*")))
    k_theory, Pk_theory = get_matter_power_spectrum(z=0, returnk=True)

    
    arg_list = []
    meta     = []  
    for run in run_dirs:
        snaps = find_all_hdf5(run)
        if not snaps:
            continue
        with h5py.File(snaps[0], 'r') as f:
            boxsize = f.attrs['boxsize']

        m = re.search(r'giant(\d+)(p3m|pm)', run)
        if not m:
            continue
        N_in  = int(re.search(r'giant(\d+)', run).group(1))
        method = m.group(2)
        ngrid = N_in if 'p3m' in run else N_in
        arg_list.append((snaps[0], boxsize, ngrid))
        meta.append((method, N_in, ngrid, boxsize))
   

    
    with mp.Pool(10) as pool:
        results = pool.starmap(compute_power_spectrum_from_snapshot, arg_list)
        
    plot_data = list(zip(meta, results))
    plot_data.sort(key=lambda x: x[0][1])   # x[0][1] er N_in
    meta_sorted, results_sorted = zip(*plot_data)
    
    import math
    n_runs = len(plot_data)
    ncols  = math.ceil(math.sqrt(n_runs))
    nrows  = math.ceil(n_runs / ncols)
    fig, axes = plt.subplots(2, 4,
                             figsize=(4*nrows, 2*ncols),
                             sharex=True, sharey=True)

    axes = axes.flatten()
    for ax, (method, N_in, ngrid, boxsize), (k, Pk) in zip(axes, meta_sorted, results_sorted):
        ax.plot(k_theory, Pk_theory, 'k-', lw=1.5)
        ax.scatter(k, Pk, s=8, alpha=0.6)
        ax.axvline(np.pi*ngrid/boxsize, color='gray', ls='--', lw=2)
        ax.set_xlim(1e-2, ( np.pi*ngrid/boxsize)* 1.1)
        ax.set_title(f"{method.upper()} (N = {ngrid}³)", fontsize=24)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.grid(which='both', ls=':')

    # 
    for ax in axes[n_runs:]:
        fig.delaxes(ax)

    # 
    fig.supxlabel(r"$k\ [\mathrm{Mpc}^{-1}]$", fontsize=24)
    fig.supylabel(r"$P(k)\ [\mathrm{Mpc}^3]$",fontsize=24)
    fig.suptitle("$P(k)$ at different N's", y=0.93, fontsize=24)

    # 
    handles, labels = axes[0].get_legend_handles_labels()

    # 
    handles.append(Line2D([0], [0], color='black', lw=1.5))
    labels .append('CLASS z=0')

    handles.append(Line2D([0], [0], marker='o', linestyle='', markersize=8, alpha=0.6))
    labels .append('CONCEPT P(k)')

    handles.append(Line2D([0], [0], color='gray', ls='--', lw=1))
    labels .append('Nyquist')

    axes[0].legend(handles, labels,
                   bbox_to_anchor = (-0.6, 1.0),
                   fontsize=16)

    plt.tight_layout()
    plt.show()


def compute_power_spectrum(delta_x, fromcic  = False, kx = 0,ky=0, kz=0, L = 0, N = 0):
    delta_k = np.fft.fftn(delta_x) 
    Pk = np.abs(delta_k)**2



    if fromcic == True:
        k_quist = np.pi * N / L
        W_CIC_corrected = (np.sinc(kx / (2 * k_quist)) * np.sinc(ky / (2 * k_quist)) * np.sinc(kz / (2 * k_quist))) ** 2
        W_CIC_corrected[W_CIC_corrected < 0.1] = 0.1  # Øg den nedre grænse
        Pk /= W_CIC_corrected

    k_vals = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)  
    # Binning af P(k), 
    k_min = 2 * np.pi / L
    k_max = k_mag.max()  # eller k_max = 2*np.pi*N/L for at få alt med
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num=300)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    Pk_binned = np.zeros_like(k_centers)
    counts = np.zeros_like(k_centers)

    #Her laver jeg en loop for at binne P(k) værdierne, jeg gør det at da de er 3D så omdanner jeg dem til 1D og binner dem. KAN VÆRE artefakter her.
    for i in range(N):
        for j in range(N):
            for k in range(N):
                k_val = k_mag[i, j, k]
                if k_val > 0:
                    bin_idx = np.digitize(k_val, k_bins) - 1
                    if 0 <= bin_idx < len(Pk_binned):
                        Pk_binned[bin_idx] += Pk[i, j, k]
                        counts[bin_idx] += 1

    # Kun divider, hvor counts > 0
    #Er faktisk lidt usikker på om det er en god ide at bruge en maske her?
    valid_bins = np.abs(counts.astype(int)) > 0
    Pk_binned[valid_bins] /= counts[valid_bins]
    
    return k_centers[valid_bins], Pk_binned[valid_bins]
