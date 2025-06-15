from classy import Class
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import scipy.interpolate as scint
import h5py
from numba import jit
# Sætter kassen op
z = 5  # Rødforskydningen, ændres globalt her!
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
c_1 = np.random.normal(0, 1/(np.sqrt(2)), size=(N, N, N))
c_2 = np.random.normal(0, 1/(np.sqrt(2)), size=(N, N, N))
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
fac = np.sqrt(2 * np.pi / (L**3)) * N**3  # Normaliseringsfaktor
delta_k = np.sqrt(P_k_grid) * fac
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



#Starter på CIC--------------------------------------------------------------------------------------------------------------------------------

#Her placerer jeg partiklerne i gitteret
particles = np.array(np.meshgrid(
    np.linspace(0, L, N, endpoint=False),
    np.linspace(0, L, N, endpoint=False),
    np.linspace(0, L, N, endpoint=False),
    indexing='ij'
)).reshape(3,-1).T  # Initial gitter af partikler


#Displacement vektorer (Det er egentlig præcis de samme som de gamle, jeg bygger dem bare igen, for at have noget at kunne sammenligne med. (Måske en dårlig ide))
Psi_kx = (1j * kx / k_mag**2) *delta_k
Psi_ky = (1j * ky / k_mag**2) *delta_k
Psi_kz = (1j * kz / k_mag**2) *delta_k

Psi_x = np.fft.ifftn(Psi_kx * R).real
Psi_y = np.fft.ifftn(Psi_ky * R).real
Psi_z = np.fft.ifftn(Psi_kz * R).real


from scipy.interpolate import RegularGridInterpolator
def periodic_interpolator(data, x_kord, y_kord, z_kord, L):
    """
    Laver en periodic interpolator af 'data' på (x_kord, y_kord, z_kord).
    """
    # Opret selve interpolations-objektet (fill_value=None, så vi ikke sætter 0 i out-of-bounds)
    interp = RegularGridInterpolator(
        (x_kord, y_kord, z_kord),
        data,
        method='linear',
        bounds_error=False,
        fill_value=None  # Vi modder, så vi aldrig kommer out-of-bounds, jeg prøver at undgå at lave en fill_value
    )

    def wrapped_interp(pts):
        # "wrap" punkter ind i [0,L) i alle dimensioner:
        pts_mod = np.mod(pts, L)
        return interp(pts_mod)
    return wrapped_interp


interp_psi_x = periodic_interpolator(Psi_x, x_kord,y_kord,z_kord,L)#
interp_psi_y = periodic_interpolator(Psi_y,x_kord,y_kord,z_kord,L)#
interp_psi_z = periodic_interpolator(Psi_z,x_kord,y_kord,z_kord,L)#

particles_mod = np.mod(particles, L)

disp_x = interp_psi_x(particles_mod)
disp_y = interp_psi_y(particles_mod)
disp_z = interp_psi_z(particles_mod)

particles += np.column_stack((
    disp_x,
    disp_y,
    disp_z
))
particles %= L



# Initialiser delta_cic gitteret
delta_cic = np.zeros((N, N, N))

# Gitterstørrelse per celle 
cell_size = L / N

for p in particles:
    # Finder den nærmeste gittercelle for hvert placeret partikel
    i = int(p[0] / cell_size) % N
    j = int(p[1] / cell_size) % N
    k = int(p[2] / cell_size) % N

    # Finder den normaliserede vægt (afstanden til gitterpunktet i hver retning)
    dx = (p[0]%cell_size) / cell_size
    dy = (p[1]%cell_size) / cell_size
    dz = (p[2]%cell_size) / cell_size

    # # Vælg en testpartikel (f.eks. partikel nr. 1000) (Det her er bare en test for at se hvor min partikel er)
    # test_idx = 1000 if particles.shape[0] > 1000 else 0
    # test_particle = particles[test_idx]

    # # Debugging: Kun udskriv for én testpartikel
    # if np.all(p == test_particle):  
    #     print(f"\nTestpartikel før fordeling:")
    #     print(f"  Position: {test_particle}")
    #     print(f"  Grid indices (i, j, k): ({i}, {j}, {k})")
    #     print(f"  Normalized distances (dx, dy, dz): ({dx:.3f}, {dy:.3f}, {dz:.3f})")

        # # Tjek hvordan vægtene bliver fordelt, bare en test her: 
        # weights = []
        # for di, dj, dk in [0, 1], [0,1], [0,1]:
        #                 weight = (1 - dx if di == 0 else dx) * \
        #                         (1 - dy if dj == 0 else dy) * \
        #                         (1 - dz if dk == 0 else dz)
        #                 weights.append(weight)
        #                 ni, nj, nk = (i + di) % N, (j + dj) % N, (k + dk) % N
        #                 print(f"  Bidrag til celle ({ni}, {nj}, {nk}): {weight:.3f}")

        #              print(f"Total vægt: {sum(weights):.3f} (burde være ≈1)\n")

    # Fordeler massen over de 8 nærmeste celler
    for di in [0, 1]:
        for dj in [0, 1]:
            for dk in [0, 1]:
                weight = ((1 - dx) if di == 0 else dx) * \
                         ((1 - dy) if dj == 0 else dy) * \
                         ((1 - dz) if dk == 0 else dz)

               
                ni, nj, nk = (i + di) % N, (j + dj) % N, (k + dk) % N  # Finder gitterkoordinater:
                delta_cic[ni, nj, nk] += weight  # Tilføj vægten til den pågældende celle



delta_cic -= 1.0

# Plotter CIC δ(x) og FFT δ(x) sammen (Lavet af chatten)--------------------------------------------------------------------------------------------------------------------------------

#Plotning af displacement felter  -----------------------------------------------------------
levels = np.linspace(np.min(delta_x[:, :, int(N/2)]), np.max(delta_x[:, :, int(N/2)]), 10)

# Opsætter subplots:
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot CIC delta(x) 
CicFig = ax[0].contourf(X[:, :, int(N/2)], Y[:, :, int(N/2)], delta_cic[:, :, int(N/2)], cmap='viridis', levels = levels)#, levels = levels)
fig.colorbar(CicFig, ax=ax[0], label=r"$\delta_{CIC}(x)$")

ax[0].set_title(rf"2D-slice af CIC $\delta(x)$ ved z={z}")
ax[0].set_xlabel("Mpc")
ax[0].set_ylabel("Mpc")

# step = 20  # Justér for at gøre quiver læsbart
# ax[0].quiver(X[::step, ::step, int(N/2)], 
#              Y[::step, ::step, int(N/2)], 
#              Psi_x[::step, ::step, int(N/2)], 
#              Psi_y[::step, ::step, int(N/2)], 
#              color='r', alpha=0.6, width = 0.010, scale = 0.1, headlength = 2, headwidth = 3)  # Justér scale

# Plotter FFT delta(x) 
FFTFig = ax[1].contourf(X[:, :, int(N/2)], Y[:, :, int(N/2)], delta_x[:, :, int(N/2)], cmap='viridis', levels = levels)#, levels = levels)
fig.colorbar(FFTFig, ax=ax[1], label=r"$\delta_{FFT}(x)$")
ax[1].set_title(rf"2D-slice af $\delta(x)$ ved z={z}")
ax[1].set_xlabel("Mpc")
ax[1].set_ylabel("Mpc")

# step = 20  # Altså de her quivers virker ikke. 
# ax[1].quiver(X[::step, ::step, int(N/2)], 
#              Y[::step, ::step, int(N/2)], 
#              psi_x[::step, ::step, int(N/2)], 
#              psi_y[::step, ::step, int(N/2)], 
#              color='r', alpha=0.6, width = 0.007, scale = 0.95, headlength = 1, headwidth = 1)  # Justér scale

plt.tight_layout()  
plt.show()

# Plotter displacement felter  -----------------------------------------------------------
def compute_power_spectrum(delta_x, fromcic  = False, kx = 0,ky=0, kz=0, L = 0, N = 0):
    V = L**3
    fac = np.sqrt(2 * np.pi / V) * N**3
    delta_k = np.fft.fftn(delta_x) 
    Pk = np.abs(delta_k)**2 /(fac**2)



    if fromcic == True:
        k_quist = np.pi * N / L
        W_CIC_corrected = (np.sinc(kx / (2*k_quist)) * np.sinc(ky / (2*k_quist)) * np.sinc(kz / (2*k_quist)))**2
        W_CIC_corrected[W_CIC_corrected == 0 ] = 1  # Øg den nedre grænse
        Pk /= (W_CIC_corrected**2)

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

k_fft, Pk_fft = compute_power_spectrum(delta_x, fromcic=False,kx=kx,ky=ky,kz=kz, L=L, N=N)
k_cic, Pk_cic = compute_power_spectrum(delta_cic, fromcic=True,kx=kx,ky=ky,kz=kz, L=L, N=N)


#Tilføjer også det teoretiske P(k) fra CLASS for at sammenligne--------------------------------------------------------------------------------------------------------------------------------
k_theory, Pk_theory = get_matter_power_spectrum(z, returnk=True)
print(np.var(delta_cic))
#Plotter powerspektrum af dem begge (Plotning er 100% gjort af Chatten)--------------------------------------------------------------------------------------------------------------------------------

plt.scatter(k_fft, Pk_fft, label='Class simulated Method', marker='o', color='blue')
plt.scatter(k_cic, Pk_cic, label='CIC Method (Corrected)', marker='o', color='orange')

# CLASS-teori som en almindelig linjeplot
plt.plot(k_theory, Pk_theory, label='CLASS z=0', linestyle='-', color='black')

vartjek=print(np.var(delta_cic))
varfft=print(np.var(delta_x))

k_ny= np.pi*N/L
plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')


# Labels og layout
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k\ \left[\mathrm{Mpc}^{-1}\right]$', fontsize=14)
plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc})^3\right]$', fontsize=14)
plt.xlim(1e-2, k_ny * 2)  # eller fx 1e-2 til 5, hvis du hellere vil sætte fast grænse
plt.title(r'Power Spectrum $P(k)$ Comparison', fontsize=16)
plt.grid(True, which='both', ls='--', alpha=0.7)
plt.legend()
plt.show()

#Debugging af CiC-----------------------------------------------------------------
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.imshow(Psi_x[:, :, N//2], origin='lower', cmap='viridis')
# plt.colorbar(label='Psi_x')
# plt.title('Psi_x (z-slice)')

# plt.subplot(1, 3, 2)
# plt.imshow(Psi_y[:, :, N//2], origin='lower', cmap='viridis')
# plt.colorbar(label='Psi_y')
# plt.title('Psi_y (z-slice)')

# plt.subplot(1, 3, 3)
# plt.imshow(Psi_z[:, :, N//2], origin='lower', cmap='viridis')
# plt.colorbar(label='Psi_z')
# plt.title('Psi_z (z-slice)')

# plt.tight_layout()
# plt.show()

# test_points = np.array([[0, 0, 0], [L/2, L/2, L/2], [L-1, L-1, L-1]])
# print("Interpolated Psi_x:", interp_psi_x(test_points))
# print("Interpolated Psi_y:", interp_psi_y(test_points))
# print("Interpolated Psi_z:", interp_psi_z(test_points))

# for p in particles[:10]:  # Test de første 10 partikler
#     dx = (p[0] % cell_size) / cell_size
#     dy = (p[1] % cell_size) / cell_size
#     dz = (p[2] % cell_size) / cell_size
#     weights = []
#     for di in [0, 1]:
#         for dj in [0, 1]:
#             for dk in [0, 1]:
#                 weight = ((1 - dx) if di == 0 else dx) * \
#                          ((1 - dy) if dj == 0 else dy) * \
#                          ((1 - dz) if dk == 0 else dz)
#                 weights.append(weight)
#     print(f"Total weight for particle {p}: {sum(weights):.3f}")