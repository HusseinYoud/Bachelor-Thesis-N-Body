import os, sys, glob, re
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numba import njit
from classy import Class
z = 0  # Rødforskydningen, ændres globalt her!
L = 500 #jeg ved ikke hvorfor den gerne vil være en float, men jeg har fundet at jeg får de bedste resultater når det er en float
N = 64  # Grid size

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
    
# Tilføj mappen med Plotter.py til sys.path:
# Finder alle .hdf5 snapshots i en given mappe
def find_all_hdf5(run_dir):
    snaps = []
    for dirpath, _, files in os.walk(run_dir):
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
    with h5py.File(snapshot_path, 'r') as f:
        coords = f['components/matter/pos'][:]

    k_vals = 2*np.pi * np.fft.fftfreq(Nmesh, d=boxsize/Nmesh)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    W = np.clip((np.sinc(kx/(2*np.pi*Nmesh/boxsize)) *
                 np.sinc(ky/(2*np.pi*Nmesh/boxsize)) *
                 np.sinc(kz/(2*np.pi*Nmesh/boxsize)))**2, 1e-3, None)

    delta = np.zeros((Nmesh,)*3, dtype=np.float64)
    delta = cic_assign(delta, coords, boxsize/Nmesh, Nmesh)
    rho_mean = coords.shape[0] / Nmesh**3
    delta = delta/rho_mean - 1.0

    delta_k = np.fft.fftn(delta)/W
    Pk = np.abs(delta_k)**2 * boxsize**3 / (Nmesh**6)

    k_min, k_max = 2*np.pi/boxsize, k_mag.max()
    if k_min <= 0 or k_max <= 0:
        return np.array([]), np.array([])
    bins = np.logspace(np.log10(k_min), np.log10(k_max), 100)
    centers = 0.5*(bins[:-1]+bins[1:])

    kf, Pf = k_mag.ravel(), Pk.ravel()
    mask = kf > 0
    counts, _ = np.histogram(kf[mask], bins=bins)
    sums, _ = np.histogram(kf[mask], bins=bins, weights=Pf[mask])

    pk_binned = np.zeros_like(centers)
    nz = counts > 0
    pk_binned[nz] = sums[nz]/counts[nz]

    return centers[nz], pk_binned[nz]


k_theory, Pk_theory = get_matter_power_spectrum(z=0, returnk=True)
# --- Main plotting ---
base_dir   = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\1vs2LPT"
sim_types  = ["PP", "PM", "P3M"]
lpt_orders = ["1LPT", "2LPT"]

# Collect data
# Collect data
data = {}
for sim in sim_types:
    data[sim] = {}
    for lpt in lpt_orders:
        run_dir = os.path.join(base_dir, sim, lpt)
        snaps   = find_all_hdf5(run_dir)
        if not snaps:
            continue
        with h5py.File(snaps[0], 'r') as f:
            coords = f['components/matter/pos'][:]
            box    = f.attrs.get('boxsize')
        Nmesh = int(round(coords.shape[0] ** (1/3)))
        k, pk = compute_power_spectrum_from_snapshot(snaps[0], box, Nmesh)
        if k.size > 0:
            data[sim][lpt] = (k, pk)

# 1) Individual scatter plots (log axes)
for sim in sim_types:
    for lpt in lpt_orders:
        if lpt in data[sim]:
            k, pk = data[sim][lpt]
            plt.figure(figsize=(6,4))
            plt.scatter(k, pk, s=10, marker='_' )
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("k [Mpc$^{-1}$]", fontsize=24)
            plt.ylabel("P(k) [Mpc$^3$]", fontsize=24)
            plt.title(f"{sim} {lpt}", fontsize=24)
            plt.grid(which="both", ls=":")
            plt.tight_layout()
            plt.close()
k_ny = np.pi * N / L  # Nyquist frequency
# 2) Overlay scatter for PM and P3M (log axes)
for sim in ["PM", "P3M"]:
    if set(lpt_orders).issubset(data[sim].keys()):
        plt.figure(figsize=(6,4))
        for lpt, marker in zip(lpt_orders, ['o','s']):
            k, pk = data[sim][lpt]
            plt.scatter(k, pk, s=10, marker=marker, label=lpt)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-2, k_ny * 1.1)
        plt.xlabel("k [Mpc$^{-1}$]", fontsize=24)
        plt.ylabel("P(k) [Mpc$^3$]", fontsize=24)
        plt.title(f"{sim}: 1 LPT vs. 2 LPT", fontsize=24)
        plt.legend()
        plt.grid(which="both", ls=":")
        plt.tight_layout()
        plt.close()

# 3) Combined overlay scatter (log axes)
plt.figure(figsize=(8,6))
k_ny = np.pi * N /L
styles = {'1LPT':'o','2LPT':'_'}
for sim in sim_types:
    for lpt in lpt_orders:
        if lpt in data[sim]:
            k, pk = data[sim][lpt]
            plt.scatter(k, pk, s=13, marker=styles[lpt], label=f"{sim} {lpt}")
plt.plot(k_theory, Pk_theory, 'k-', lw=1.5, label='CLASS z = 0')
plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
plt.xscale('log')
plt.yscale('log')
ticks = [1e-2, 1e-1, 1e0]
plt.xlim(1e-2, k_ny * 2)
plt.xlabel("k [Mpc$^{-1}$]", fontsize=24)
plt.ylabel("P(k) [Mpc$^3$]", fontsize=24)
tick_labels = [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
plt.xticks(ticks, tick_labels, fontsize=20)

yticks = [1e2, 1e3, 1e4, 1e5]
yticklabels = [r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$"]
plt.yticks(yticks, yticklabels, fontsize=20)
plt.title("PP, PM, P3M: All $P(k)$, N = $ 64^3 $, L = 500 [$ Mpc $]" , fontsize=24)
plt.legend(fontsize=24, ncol=2)
plt.grid(which="both", ls=":")
plt.tight_layout()
plt.show()

print("Tilgængelige LPT-ordener for PP:", data["PP"].keys())



snaps_pp1 = find_all_hdf5(os.path.join(base_dir, "PP", "1LPT"))
import matplotlib.pyplot as plt
snaps_pp1 = find_all_hdf5(os.path.join(base_dir, "PP", "1LPT"))
sim = "PP"
lpt_orders = ["1LPT", "2LPT"]

plt.figure(figsize=(6,4))
for lpt in lpt_orders:
    if lpt in data[sim]:
        k, pk = data[sim][lpt]
        plt.scatter(k, pk, s=10, marker="_", label=f"{sim} {lpt}")
    else:
        print(f"*** ADVARSEL: Ingen data fundet for {sim} {lpt} ***")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("k [Mpc$^{-1}$]", fontsize=14)
plt.ylabel("P(k) [Mpc$^3$]", fontsize=14)
plt.title(f"{sim}: 1 LPT vs. 2 LPT", fontsize=16)
plt.legend()
plt.grid(which="both", ls=":")
plt.tight_layout()
plt.close()

# Efter det store "for sim in sim_types: for lpt in lpt_orders:"-loop:
# for sim in ["PP"]:
#     for lpt in ["1LPT", "2LPT"]:
#         if lpt not in data[sim]:
#             print(f">>> FEJL: Ingen data i data['{sim}']['{lpt}']")
#         else:
#             k_arr, pk_arr = data[sim][lpt]
#             print(f"{sim} {lpt}: Antal bølge­tal = {len(k_arr)}, første k = {k_arr[0]:.3e}, sidste k = {k_arr[-1]:.3e}")

# # Efter dit normale loop, hvor du looper over sim_types og lpt_orders:
# for sim in ["PP"]:
#     for lpt in ["1LPT", "2LPT"]:
#         if lpt not in data[sim]:
#             print(f">>> Ingen data for {sim} {lpt}")
#         else:
#             k_arr, pk_arr = data[sim][lpt]
#             print(f"{sim} {lpt}: Antal bølgetal = {len(k_arr)}, k_min = {k_arr[0]:.3e}, k_max = {k_arr[-1]:.3e}")
#             # Print et uddrag af selve pk-værdierne:
#             print(f"   Første 5 P(k)-værdier: {pk_arr[:5]}")
#             print(f"   Sidste 5 P(k)-værdier: {pk_arr[-5:]}")
# import os

# base = base_dir  # det du bruger i scriptet
# print("Indhold af PP/1LPT-mappen:", os.listdir(os.path.join(base, "PP", "1LPT")))
# print("Indhold af PP/2LPT-mappen:", os.listdir(os.path.join(base, "PP", "2LPT")))

# k1, pk1 = data["PP"]["1LPT"]
# k2, pk2 = data["PP"]["2LPT"]

# # Beregn relativ forskel
# rel_forskel = (pk2 - pk1) / pk1

# print("Maksimal absolut relativ forskel:", np.max(np.abs(rel_forskel)))
# print("Relativ forskel for de første 5 punkter:", rel_forskel[:5])
# print("Relativ forskel for de sidste 5 punkter:", rel_forskel[-5:])



