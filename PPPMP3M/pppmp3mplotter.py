# pp_pm_p3m_plotter.py
import numpy as np
import h5py
import matplotlib.pyplot as plt
from classy import Class
from numba import njit
import os, time, glob
import re
# ---------- Brugervalg ----------
z        = 0                 # rødforskydning
L        = 500.0             # boksstørrelse [Mpc/h]
N        = 64                # grid (antager N^3 partikler)
base_dir = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\PMP3MPP"  # <-- RET TIL DIN MAPPE
sim_types = ["PP", "PM", "P3M"]
use_corr = False
# --------------------------------

runtime = {
    "PP" : "2 days, 23:27:37",
    "PM" : "12.1 s",
    "P3M": "4:18 min",
}

def parse_runtime(s):
    """'2 days, 23:27:37' -> sek; '4:18 min' -> sek; '12.1 s' -> sek"""
    s = s.strip().lower()
    # dage?
    days, rest = 0, s
    m = re.match(r"(\d+)\s*day", s)
    if m:
        days = int(m.group(1))
        rest = s.split(",",1)[1] if "," in s else "0:0:0"

    # minutter-format?
    if "min" in rest:
        h, m = 0, 0
        mm_ss = rest.split("min")[0].strip()
        if ":" in mm_ss:
            m, s = map(float, mm_ss.split(":"))
        else:
            m = float(mm_ss)
            s = 0
        return days*86400 + h*3600 + m*60 + s

    # timer: minutter: sekunder format
    if ":" in rest:
        parts = list(map(float, rest.split(":")))
        while len(parts) < 3:         #  mm:ss  ->  0:mm:ss
            parts.insert(0,0)
        h, m, s = parts
        return days*86400 + h*3600 + m*60 + s

    # ellers '12.1 s'
    if "s" in rest:
        sec = float(rest.split("s")[0])
        return days*86400 + sec

    raise ValueError(f"Uforståeligt runtime-format: {s}")

# ---------- CLASS ----------
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

    # #Jeg er lidt usikker på hvorfor vi skal bruge den her vækstfaktor, men chatten siger det er vigtigt
    # D_z = cosmo.scale_independent_growth_factor(z)
    # Pk *= D_z**2  # Skaler P(k) korrekt
    #Nvm jeg fandt ud af at det er bare sygt forkert. >:(

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




k_theory, Pk_theory = get_matter_power_spectrum(z, returnk=True)
kNy = np.pi*N/L

# ---------- Hjælpefunktioner ----------
def find_ps_file(folder):
    """
    Returnér første powerspectrum-fil i 'folder' eller dets
    undermapper. Ignorerer mapper.
    """
    pattern = os.path.join(folder, "**", "powerspec*")
    candidates = [f for f in glob.glob(pattern, recursive=True)
                  if os.path.isfile(f)]
    return sorted(candidates)[0] if candidates else None

def load_concept_ps(filepath, corrected=True, enc='utf-8'):
    """
    Returnerer tre arrays:
      k       – bølgetal
      P_sim   – valgt spektrum (korrigeret eller rå)
      P_lin   – lineært spektrum fra filens 5. kolonne
    """
    with open(filepath, 'r', encoding=enc, errors='ignore') as fh:
        k, P_raw, P_corr, P_lin = np.loadtxt(
            fh, comments='#', usecols=(0, 2, 3, 4), unpack=True)

    P_sim = P_corr if corrected else P_raw
    return k, P_sim, P_lin

# ---------- data-loop ----------
kNy = np.pi * N / L

data_pk   = {}   # (k, P_sim)
err_dist  = {}   # fejl pr. k
acc_mean  = {}   # middel­fejl
run_sec   = {}   # runtime i sek

for sim in sim_types:
    ps_file = find_ps_file(os.path.join(base_dir, sim))
    if ps_file is None:
        print(f"[!] ingen powerspec-fil fundet i {sim}")
        continue

    k_sim, P_sim, _ = load_concept_ps(ps_file, use_corr)
    data_pk[sim] = (k_sim, P_sim)

    mask = k_sim <= kNy
    Pk_interp = np.interp(k_sim[mask], k_theory, Pk_theory)
    rel_err = np.abs(P_sim[mask] - Pk_interp) / Pk_interp
    err_dist[sim] = rel_err
    acc_mean[sim] = rel_err.mean()
    run_sec[sim]  = parse_runtime(runtime[sim])

    print(f"{sim:3s}: {len(k_sim)} k-bins | mean-err = {acc_mean[sim]:.3f} | runtime = {run_sec[sim]:.1f} s")

# ---------- Figur 1 – P(k) mod CLASS ----------
plt.figure(figsize=(8,8))
markers = dict(PP='o', PM='s', P3M='^')
for sim,(k,P) in data_pk.items():
    plt.scatter(k, P, marker=markers[sim], s=14, label=f"{sim}.00")
plt.plot(k_theory, Pk_theory, 'k-', lw=1.8, label=f'CLASS  z={z}')
plt.axvline(kNy, ls='--', color='grey', label='Nyquist')
plt.xscale('log'); plt.yscale('log')
plt.xlim(1e-2,kNy)
plt.xlabel(r'$k\,[\, \mathrm{Mpc}^{-1}]$', fontsize=24)
plt.ylabel(r'$P(k)\,[(\mathrm{Mpc})^3]$', fontsize=24)
plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.title(f'$P(k)$ =  PP / PM / P3M   ($N={N}^3$, $L={L}$ Mpc)', fontsize=24)
plt.grid(ls=':'); plt.legend(fontsize=24); plt.tight_layout()



# ---------- Figur 3 – fejl-histogrammer ----------
fig, axs = plt.subplots(1, 3, figsize=(12, 8), sharey=True, sharex=True)

for ax, sim, col in zip(
        axs, ["PP", "PM", "P3M"], ['tab:blue', 'tab:orange', 'tab:green']):
    if sim not in data_pk:
        ax.axis('off')
        continue

    k_sim, P_sim = data_pk[sim]

    # only up to Nyquist
    mask       = k_sim <= kNy
    k_vals     = k_sim[mask]
    Pk_interp  = np.interp(k_vals, k_theory, Pk_theory)
    rel_dev_pc = 100.0 * (P_sim[mask] - Pk_interp) / Pk_interp   # signed %

    # bar plot (one bar per k-bin)
    ax.bar(k_vals, rel_dev_pc,
           width=np.diff(np.log10(k_vals)).min()*0.2,  # narrow logarithmic bars
           color=col, alpha=0.7, align='center')

    ax.axhline(0, color='k', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_yticks(np.arange(0, 260, 5), minor=True)  
    ax.set_title(sim, fontsize=24)
    ax.grid(ls=':', alpha=0.7)

axs[0].set_ylabel('Relative deviation  [%]', fontsize=24)
fig.suptitle('Deviation from linear power spectrum', fontsize=24)
plt.show()
plt.tight_layout()