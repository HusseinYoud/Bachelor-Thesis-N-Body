"""
This script is meant to compare the CLASS power spectrum with the one obtained from CONCEPT. As well as compare the powerspectrum obtained directly from CONCEPT (i refer to thesse as .00 files) with the ones made from the snapshots (which i refer to as .hdf5 files)  
Throughout the script a chocie between using the fac factor (volumen normalisation) or not. For all CONCEPT .hdf5 files the fac factor is not used unless that options is turned True. 
For testing purposes a choice between using the deconvolution or not is also available. 

"""
from classy import Class
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.interpolate import RegularGridInterpolator #Used for the Cloud in Cell interpolation later on
from matplotlib.ticker import LogFormatterMathtext
import scipy.interpolate as scint
from numba import jit #imported for the jit decorator to speed up the cic_density function as well as the particle_assigment function. Outcomment if not needed/wanted
# Builds the CLASS power spectrum
z1 = 0
z2 = 5  
L = 1000
N = 64  
def get_matter_power_spectrum(z, k = np.logspace(-4, 0, 500), returnk=False, plot = False):
    """
    Computes the linear matter power spectrum using CLASS.
    """
    params = {
        'output': 'mPk',
        'H0': 67.36,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'Omega_k': 0.0,
        'n_s': 0.9649,
        'A_s': 2.105e-9,
        'z_pk': z,  # Uses the global redshift, which is either 0 or 5
        'P_k_max_h/Mpc': 10,
        'non linear': 'none',     
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    k_vals = k.flatten()

    Pk = np.array([cosmo.pk(ka, z) for ka in k_vals])

 # Plots P(k) if plot is True
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
        return k_vals, Pk 
    else:
        return Pk

k_ny= np.pi*N/L #Nyquist frequency

#Reading in the CONCEPT hdf5 snapshot files
snappath = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Debugging og had\snapshots" #Path to the CONCEPT hdf5 files

#Reads the positions of the particles in the CONCEPT snapshot files
def read_positions(filename):
    """
    Digs in the CONCEPT hdf5 file and reads only the positions of the particles.
    """
    fullpath = os.path.join(snappath, filename)
    with h5py.File(fullpath, 'r') as f:
        return f['components/matter/pos'][:]

#defines the window function
def cic_window_ft(N, L):
    d   = L/N
    k_vals= 2*np.pi*np.fft.fftfreq(N, d=d)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_quist = np.pi*(d)**(-1)
    Wx = np.sinc(kx/(2*k_quist))
    Wy = np.sinc(ky/(2*k_quist))
    Wz = np.sinc(kz/(2*k_quist))
    W = (Wx*Wy*Wz)**2
    #print(W)
    return W

#Defines the denconvulution function:
def deconvolve_cic(delta_grid, L, N):
    delta_k = np.fft.fftn(delta_grid, norm = 'ortho') #Norm = 'ortho' is used to ensure that the FFT is normalized, is scaled with (1/sqrt(n)) https://numpy.org/doc/stable/reference/routines.fft.html
    W  = cic_window_ft(N, L)
    delta_k /= W
    return np.fft.ifftn(delta_k, norm = 'ortho').real

#Defines the CiC grid mass assignment:
@jit #outcomment if not needed/wanted
def cic_density(positions, L, N):
    """
    Mass assignment to a grid using Cloud-in-Cell (CIC) method. The three giant for-loops are computationally expensive, if nessecary a jit decorator can be used to speed up the process. https://numba.pydata.org/numba-doc/dev/user/jit.html
    just add a call at the top of the script: from numba import jit and then add @jit to the function cic_density
    The function takes the positions of the particles and places them in a grid.  
    """
    delta_cic = np.zeros((N, N, N), dtype=np.float64)  # Places mass in the grid
    cell_size = L / N  # Size of each grid cell

    for p in positions:
        i = int(p[0] / cell_size) % N
        j = int(p[1] / cell_size) % N
        k = int(p[2] / cell_size) % N

        # Normalised weight for each cell
        dx = (p[0]%cell_size) / cell_size
        dy = (p[1]%cell_size) / cell_size
        dz = (p[2]%cell_size) / cell_size

        # Iterate over the 8 surrounding cells and distributes the mass
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    weight = ((1 - dx) if di == 0 else dx) * \
                            ((1 - dy) if dj == 0 else dy) * \
                            ((1 - dz) if dk == 0 else dz)

                
                    ni, nj, nk = (i + di) % N, (j + dj) % N, (k + dk) % N  # Finds the index of the cell
                    delta_cic[ni, nj, nk] += weight  # Adds the weight to the cell

    return delta_cic 

#Computes the power spectrum from the hdf5 snapshot files
def compute_power_spectrum_from_snapshot(snapshot_path, L, Nmesh=N, deconvolve=False, use_fac = False):
    """
    Computes the power spectrum from a CONCEPT snapshot file. Both options for deconvolution and fac are available. If fac is not used a simple -1 to account for overdensity is used.
    Though at this point both use_fac = True and False yield the same result. Depending on the box_size and the number of particles this should not be the case in the future.
    The function takes the path to the snapshot file, the boxsize L, the number of particles N, and the options for deconvolution and fac.
    """

    # load positions
    with h5py.File(snapshot_path, 'r') as f:
        pos = f['components/matter/pos'][:]     # shape (N, 3)
    # print("pos.shape =", pos.shape[0]) #Sanity check to check the number of particles from hdf5 file
    # print("Nmesh^3=" , Nmesh**3) #does it match the number of particles in the hdf5 file?
    # builds CIC density grid
    delta = np.zeros((Nmesh,)*3, dtype=np.float32)

    delta = cic_density(pos, L, Nmesh)

    # convert to overdensity 
    if use_fac == False:
        cell_vol  = (L/Nmesh)**3
        rho_cell  = delta/(cell_vol)
        rho_mean = pos.shape[0] / (L**3)
        delta = (rho_cell/rho_mean) - 1.0
    else:
            delta -= 1

    # optionally deconvolve before FFT
    if deconvolve == True:
        delta = deconvolve_cic(delta, L, Nmesh)


    delta_k = np.fft.fftn(delta) 
    

    V = L**3
    if use_fac == True:
        # fac volumens convention
        fac  = Nmesh**3 * np.sqrt(2*np.pi / V)
        Pk3d = np.abs(delta_k)**2 /(fac**2)
    else:
        # The grid convention
        Pk3d = np.abs(delta_k)**2 * V / (Nmesh**6)

    # build k‐grid & flatten
    k_vals = 2 * np.pi * np.fft.fftfreq(Nmesh, d=L/Nmesh)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2).ravel()

    Pk_flat = Pk3d.ravel()
    mask = k_mag > 0
    k_mag = k_mag[mask]
    Pk_flat = Pk_flat[mask]

    # binning into logarithmic bins
    k_min = 2*np.pi / L
    k_max = k_mag.max()
    kbins = np.logspace(np.log10(k_min), np.log10(k_max), 300)
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])
    counts,   _ = np.histogram(k_mag, bins=kbins)
    Psum,     _ = np.histogram(k_mag, bins=kbins, weights=Pk_flat)

    valid = counts > 0
    Pk_binned = np.zeros_like(k_centers)
    Pk_binned[valid] = Psum[valid] / counts[valid]

    return k_centers[valid], Pk_binned[valid]

#Loading in the files (snapshot files (.hdf5))
file0hdf5 = 'snapshot_a=0.17.hdf5'   # Snapshot at z = 5
file1hdf5 = 'snapshot_a=1.00.hdf5'   # Snapshot at z = 0

#Loading in the entire file path
full0 = os.path.join(snappath, file0hdf5)
full1 = os.path.join(snappath, file1hdf5)

#Creates the powerspectrum from the CONCEPT snapshot files using the compute_power_spectrum_from_snapshot function
def create_snapshotPS(full0, full1, L, N):
    #no deconvulution without the fac factor
    k0_nodeconv, p0_nodeconv= compute_power_spectrum_from_snapshot(full0, L, N, deconvolve=False,use_fac=False)
    k1_nodeconv, p1_nodeconv= compute_power_spectrum_from_snapshot(full1, L, N, deconvolve=False,use_fac=False)

    #with deconvulution without the fac factor
    k0_deconv, p0_deconv= compute_power_spectrum_from_snapshot(full0, L, N, deconvolve=True,use_fac=False)
    k1_deconv, p1_deconv= compute_power_spectrum_from_snapshot(full1, L, N, deconvolve=True,use_fac=False)
    #no deconvulution with the fac factor
    k0_nodeconvfac, p0_nodeconvfac= compute_power_spectrum_from_snapshot(full0, L, N, deconvolve=False,use_fac=True)
    k1_nodeconvfac, p1_nodeconvfac= compute_power_spectrum_from_snapshot(full1, L, N, deconvolve=False,use_fac=True)

    #with deconvulution with the fac factor
    k0_deconvfac, p0_deconvfac= compute_power_spectrum_from_snapshot(full0, L, N, deconvolve=True,use_fac=True)
    k1_deconvfac, p1_deconvfac= compute_power_spectrum_from_snapshot(full1, L, N, deconvolve=True,use_fac=True)

    return (        k0_nodeconv, p0_nodeconv,
        k1_nodeconv, p1_nodeconv,
        k0_deconv,    p0_deconv,
        k1_deconv,    p1_deconv,
        k0_nodeconvfac, p0_nodeconvfac,
        k1_nodeconvfac, p1_nodeconvfac,
        k0_deconvfac,   p0_deconvfac,
        k1_deconvfac,   p1_deconvfac)
snap_results = create_snapshotPS(full0, full1, L, N)

def plot_snapshotPS(snap_results, plot = False,nodeconv = False, usefac = False):
    if plot == False:
        return
    (
    k0_nodeconv, p0_nodeconv,
    k1_nodeconv, p1_nodeconv,
    k0_deconv,    p0_deconv,
    k1_deconv,    p1_deconv,
    k0_nodeconvfac, p0_nodeconvfac,
    k1_nodeconvfac, p1_nodeconvfac,
    k0_deconvfac,   p0_deconvfac,
    k1_deconvfac,   p1_deconvfac
    ) = snap_results
    k_ny= np.pi*N/L #Nyquist frequency

    if nodeconv == True:
        plt.scatter(k0_nodeconv, p0_nodeconv, alpha=0.5, label='CONCEPT snap z=5 no deconv no fac', color='purple',marker='|')
        plt.scatter(k1_nodeconv, p1_nodeconv, alpha=0.5, label='CONCEPT snap z=0 no deconv no fac', color='blue',marker='|')
        if usefac == True:
            plt.scatter(k0_nodeconvfac, p0_nodeconvfac, alpha=0.5, label='CONCEPT snap z=5 no deconv with fac', color='green',marker='|')
            plt.scatter(k1_nodeconvfac, p1_nodeconvfac, alpha=0.5, label='CONCEPT snap z=0 no deconv with fac', color='brown',marker='|')
    if usefac == True:
        plt.scatter(k0_deconvfac, p0_deconvfac, alpha=0.5, label='CONCEPT snap z=5 deconv with fac', color='yellow',marker='_')
        plt.scatter(k1_deconvfac, p1_deconvfac, alpha=0.5, label='CONCEPT snap z=0 deconv with fac', color='olive',marker='_')

    plt.scatter(k0_deconv, p0_deconv, alpha=0.5, label='CONCEPT snap z=5 deconv no fac', color='red',marker='_')
    plt.scatter(k1_deconv, p1_deconv, alpha=0.5, label='CONCEPT snap z=0 deconv no fac', color='orange',marker='_')


    plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k\ \left[1/\mathrm{Mpc}\right]$', fontsize=14)
    plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc})\right]$', fontsize=14)
    plt.xlim(1e-2, k_ny * 1.1)  
    plt.title(r'Power Spectrum $P(k)$ exlusively from snapshots', fontsize=16)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.show()
plot_snapshotPS(snap_results, plot=False, nodeconv=True, usefac=True)

#Plotting the CLASS simulated powerspectrum as well as the CiC powerspectrum to compare with deconvolution---------------
def CLASS_simualtor(L,N, z):
    x_kord = np.linspace(0, L, N, endpoint=False)  
    y_kord = np.linspace(0, L, N, endpoint=False)  
    z_kord = np.linspace(0, L, N, endpoint=False)  
    X, Y, Z = np.meshgrid(x_kord, y_kord, z_kord, indexing ='ij')    

    k_vals = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing ='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0, 0, 0] = 1e-10  

    # cosmo = Class()
    # cosmo.set({'Omega_b':0.0486,'Omega_cdm':0.2589,'h':0.6774,
    #            'A_s':2.142e-9,'n_s':0.9667,'z_pk':z,'k_max_1/Mpc':k_mag.max()})
    # cosmo.compute()
    k_class, Pk_class = get_matter_power_spectrum(z, k = k_mag, returnk=True) 
    pk_interp = scint.interp1d(k_class, Pk_class, bounds_error=False, fill_value="extrapolate") 

    k_mag_clipped = np.clip(k_mag, k_class.min(), k_class.max())  
    P_k_grid = pk_interp(k_mag_clipped) 


    c_1 = np.random.normal(0, 1/np.sqrt(2), size=(N, N, N))
    c_2 = np.random.normal(0, 1/np.sqrt(2), size=(N, N, N))
    R = c_1 + 1j * c_2  
    R[0,0,0] = R[0,0,0].real  
    for i in np.arange(N):
        for j in np.arange(N):
            for k in np.arange(N):
                i_sym = (-i)%N
                j_sym = (-j)%N
                k_sym = (-k)%N
                if (i,j,k) != (i_sym,j_sym,k_sym):
                    R[i_sym,j_sym,k_sym] = np.conj(R[i,j,k])



    V = L**3
    fac = np.sqrt(2 * np.pi / V) * N**3
    delta_k = np.sqrt(P_k_grid)*fac 
    delta_x = np.fft.ifftn(delta_k * R)  
    delta_x = delta_x.real  


    psi_kx = 1j * kx /k_mag**2 * delta_k 
    psi_ky = 1j * ky /k_mag**2 * delta_k 
    psi_kz = 1j * kz /k_mag**2 * delta_k 


    psi_x = np.fft.ifftn(psi_kx * R).real
    psi_y = np.fft.ifftn(psi_ky * R).real
    psi_z = np.fft.ifftn(psi_kz * R).real
    return delta_x, (psi_x, psi_y, psi_z), (kx,ky,kz)


#Places the particles in the grid and uses the CIC method to assign them to the grid: Standard CIC method used on CLASS simulated data
@jit(forceobj=True) #outcomment if not needed/wanted
def particle_assigment(psi, L,N):
    """
    This function takes the displace vectors psi (which from CLASS simulator earlier is just a tuple of the three vector components), the boxsize L and the number of particles N.
    It works in the same way as the earlier seen cic_density function, but now uses the displace vectors to assign the particles to the grid.
    Once again the three for loops in the bottom of the function are computationally expensive, but can be sped up with the jit decorator from numba.
    """
    psi_x, psi_y, psi_z = psi
    x_kord = np.linspace(0, L, N, endpoint=False)  
    y_kord = np.linspace(0, L, N, endpoint=False)  
    z_kord = np.linspace(0, L, N, endpoint=False)  

    particles = np.array(np.meshgrid(
        np.linspace(0, L, N, endpoint=False),
        np.linspace(0, L, N, endpoint=False),
        np.linspace(0, L, N, endpoint=False),
        indexing='ij'
    )).reshape(3,-1).T



    def periodic_interpolator(data, x_kord, y_kord, z_kord, L):
        """
        Creates a periodic interpolator for the given data on a 3D grid.
        The data is assumed to be defined on a grid with periodic boundary conditions.
        """

        interp = RegularGridInterpolator(
            (x_kord, y_kord, z_kord),
            data,
            method='linear',
            bounds_error=False,
            fill_value=None  
        )

        def wrapped_interp(pts):
            pts_mod = np.mod(pts, L)
            return interp(pts_mod)
        return wrapped_interp


    interp_psi_x = periodic_interpolator(psi_x, x_kord,y_kord,z_kord,L)
    interp_psi_y = periodic_interpolator(psi_y,x_kord,y_kord,z_kord,L)
    interp_psi_z = periodic_interpolator(psi_z,x_kord,y_kord,z_kord,L)

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

    delta_cic = np.zeros((N, N, N))

    cell_size = L / N

    for p in particles:

        i = int(p[0] / cell_size) % N
        j = int(p[1] / cell_size) % N
        k = int(p[2] / cell_size) % N


        dx = (p[0]%cell_size) / cell_size
        dy = (p[1]%cell_size) / cell_size
        dz = (p[2]%cell_size) / cell_size

        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    weight = ((1 - dx) if di == 0 else dx) * \
                            ((1 - dy) if dj == 0 else dy) * \
                            ((1 - dz) if dk == 0 else dz)

                
                    ni, nj, nk = (i + di) % N, (j + dj) % N, (k + dk) % N 
                    delta_cic[ni, nj, nk] += weight  
    return delta_cic - 1.0 


#Finds the powerspectrum for both the CIC and the FFT method 
def compute_power_spectrum(delta_x, fromcic  = False, kx = 0,ky=0, kz=0, L = 0, N = 0, withfac = False):
    """
    Computes the power spectrum from the density field delta_x using FFT. 
    The function takes the density field, the boxsize L, the number of particles N, and the option to use the fac factor.
    """
    delta_k = np.fft.fftn(delta_x) 

    if withfac == True:
        V = L**3
        fac = np.sqrt(2 * np.pi / V) * N**3
        Pk = np.abs(delta_k)**2 /fac**2
    else:
        Pk = np.abs(delta_k)**2

    if fromcic == True:
        k_quist = np.pi * N / L
        W_CIC_corrected = (np.sinc(kx / (2 * k_quist)) * np.sinc(ky / (2 * k_quist)) * np.sinc(kz / (2 * k_quist)))**2
        #W_CIC_corrected[W_CIC_corrected < 0.1] = 0.1  #comment back in if wanted
        Pk /= (W_CIC_corrected**2)

    k_vals = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)  

    k_min = 2 * np.pi / L
    k_max = k_mag.max() 
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num=300)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    Pk_binned = np.zeros_like(k_centers)
    counts = np.zeros_like(k_centers)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                k_val = k_mag[i, j, k]
                if k_val > 0:
                    bin_idx = np.digitize(k_val, k_bins) - 1
                    if 0 <= bin_idx < len(Pk_binned):
                        Pk_binned[bin_idx] += Pk[i, j, k]
                        counts[bin_idx] += 1
    valid_bins = np.abs(counts.astype(int)) > 0
    Pk_binned[valid_bins] /= counts[valid_bins]
    
    return k_centers[valid_bins], Pk_binned[valid_bins]

delta_x1 ,psi1, (kx1,ky1,kz1) = CLASS_simualtor(L,N,z1)
delta_cic = particle_assigment(psi1, L, N)
k_fft, Pk_fft = compute_power_spectrum(delta_x1, fromcic=False,kx=kx1,ky=ky1,kz=kz1, L=L, N=N,withfac=False)
k_cic, Pk_cic = compute_power_spectrum(delta_cic, fromcic=True,kx=kx1,ky=ky1,kz=kz1, L=L, N=N,withfac=False)

delta_x2z5 ,psi2, (kx2,ky2,kz2) = CLASS_simualtor(L,N,z2)
delta_cicz5 = particle_assigment(psi2, L, N)
k_fftz5, Pk_fftz5 = compute_power_spectrum(delta_x2z5, fromcic=False,kx=kx2,ky=ky2,kz=kz2, L=L, N=N,withfac=False)
k_cicz5, Pk_cicz5 = compute_power_spectrum(delta_cicz5, fromcic=True,kx=kx2,ky=ky2,kz=kz2, L=L, N=N,withfac=False)


delta_x5 ,psi5, (kx5,ky5,kz5) = CLASS_simualtor(L,N,z1)
delta_cic2 = particle_assigment(psi5, L, N)
k_fftfac, Pk_fftfac = compute_power_spectrum(delta_x5, fromcic=False,kx=kx5,ky=ky5,kz=kz5, L=L, N=N,withfac=True)
k_cicfac, Pk_cicfac = compute_power_spectrum(delta_cic2, fromcic=True,kx=kx5,ky=ky5,kz=kz5, L=L, N=N,withfac=True)

delta_x6z5 ,psi6, (kx6,ky6,kz6) = CLASS_simualtor(L,N,z2)
delta_cicz52 = particle_assigment(psi6, L, N)
k_fftz5fac, Pk_fftz5fac = compute_power_spectrum(delta_x6z5, fromcic=False,kx=kx6,ky=ky6,kz=kz6, L=L, N=N,withfac=True)
k_cicz5fac, Pk_cicz5fac = compute_power_spectrum(delta_cicz52, fromcic=True,kx=kx6,ky=ky6,kz=kz6, L=L, N=N,withfac=True)

#Creates powerspectrum from CLASS at different redshifts-------------------------------------------------------------------------------
k_theory, Pk_theory = get_matter_power_spectrum(z1, returnk=True)
k_theory5, Pk_theory5 = get_matter_power_spectrum(z2, returnk=True)

#Plotting the CLASS powerspectrum------------------------------------------------------------------------------------------------------
def CLASS_plot(plot = False):
    if plot == False:
        return
    
    plt.plot(k_theory, Pk_theory, label='CLASS z = 0', linestyle='-', color='black')
    plt.plot(k_theory5, Pk_theory5, label='CLASS z = 5', linestyle='-', color='orange')


    plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k\ \left[1/\mathrm{Mpc}\right]$', fontsize=14)
    plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc})\right]$', fontsize=14)
    plt.xlim(1e-2, k_ny*1.1)  # eller fx 1e-2 til 5, hvis du hellere vil sætte fast grænse
    plt.title(r'Power Spectrum $P(k)$ exlusively from CLASS', fontsize=16)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.show()
CLASS_plot(plot=False)

#Testing this with the fac (volumen normalisation) convention on CONCEPTs HDF5 files vs CLASS--------------------------------------------------------------------------------
def conventionCONCLASS(snap_results, plot=False, nodeconv = False, usefac = False):
    (
    k0_nodeconv, p0_nodeconv,
    k1_nodeconv, p1_nodeconv,
    k0_deconv,    p0_deconv,
    k1_deconv,    p1_deconv,
    k0_nodeconvfac, p0_nodeconvfac,
    k1_nodeconvfac, p1_nodeconvfac,
    k0_deconvfac,   p0_deconvfac,
    k1_deconvfac,   p1_deconvfac
    ) = snap_results

   
    if plot == False:
        return
    
    if nodeconv == True:
        plt.scatter(k0_nodeconv, p0_nodeconv, alpha=0.5, label='CONCEPT snap z=5 no deconv no fac', color='purple',marker='|')

        plt.scatter(k1_nodeconv, p1_nodeconv, alpha=0.5, label='CONCEPT snap z=0 no deconv no fac', color='blue',marker='|')

        if usefac == True:
                plt.scatter(k0_nodeconvfac, p0_nodeconvfac, alpha=0.5, label='CONCEPT snap z=5 no deconv with fac', color='green',marker='|')
                plt.scatter(k1_nodeconvfac, p1_nodeconvfac, alpha=0.5, label='CONCEPT snap z=0 no deconv with fac', color='brown',marker='|')

    if usefac == True:
        plt.scatter(k0_deconvfac, p0_deconvfac, alpha=0.5, label='CONCEPT snap z=5 deconv with fac', color='yellow',marker='_')
        plt.scatter(k1_deconvfac, p1_deconvfac, alpha=0.5, label='CONCEPT snap z=0 deconv with fac', color='olive',marker='_')

    plt.scatter(k0_deconv, p0_deconv, alpha=0.5, label='CONCEPT snap z=5 deconv no fac', color='red',marker='_')
    plt.scatter(k1_deconv, p1_deconv, alpha=0.5, label='CONCEPT snap z=0 deconv no fac', color='orange',marker='_')

    plt.plot(k_theory, Pk_theory, label='CLASS z = 0', linestyle='-', color='black')
    plt.plot(k_theory5, Pk_theory5, label='CLASS z = 5', linestyle='-', color='orange')

    plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k\ \left[1/\mathrm{Mpc}\right]$', fontsize=14)
    plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc})\right]$', fontsize=14)
    plt.xlim(1e-2, k_ny * 1.1)  # Change x limits here.
    plt.title(r'Tests of the ', fontsize=16)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.show()
conventionCONCLASS(snap_results,plot=False, nodeconv=False, usefac=True)

#Testing with own CiC and the CLASS powerspectrum with and without the fac factor vs concepts-----------------------------------------------
def CiCvsSnap(snap_results, plot=False, nodeconv = False, usefac = False, cicnofac= False):
    (
    k0_nodeconv, p0_nodeconv,
    k1_nodeconv, p1_nodeconv,
    k0_deconv,    p0_deconv,
    k1_deconv,    p1_deconv,
    k0_nodeconvfac, p0_nodeconvfac,
    k1_nodeconvfac, p1_nodeconvfac,
    k0_deconvfac,   p0_deconvfac,
    k1_deconvfac,   p1_deconvfac
    ) = snap_results

    if plot == False:
        return
    
    if nodeconv == True:
        plt.scatter(k0_nodeconv, p0_nodeconv, alpha=0.5, label='CONCEPT snap z=5 no deconv no fac', color='purple',marker='|')
        plt.scatter(k1_nodeconv, p1_nodeconv, alpha=0.5, label='CONCEPT snap z=0 no deconv no fac', color='blue',marker='|')
        
        if usefac == True:
            plt.scatter(k0_nodeconvfac, p0_nodeconvfac, alpha=0.5, label='CONCEPT snap z=5 no deconv with fac', color='green',marker='|')
            plt.scatter(k1_nodeconvfac, p1_nodeconvfac, alpha=0.5, label='CONCEPT snap z=0 no deconv with fac', color='brown',marker='|')

    if usefac == True:
        plt.scatter(k1_deconvfac, p1_deconvfac, alpha=0.5, label='CONCEPT snap z=0 deconv with fac', color='olive',marker='_')
        plt.scatter(k0_deconvfac, p0_deconvfac, alpha=0.5, label='CONCEPT snap z=5 deconv with fac', color='yellow',marker='_')
   
    if cicnofac == True:
        plt.scatter(k_fft, Pk_fft , label='Class simulated Method z=0', marker='o', color='mediumblue')
        plt.scatter(k_cic, Pk_cic , label='CIC Method (Corrected) z=0', marker='o', color='orange')
        plt.scatter(k_fftz5, Pk_fftz5, label='Class simulated Method z=5', marker='o', color='lime')
        plt.scatter(k_cicz5, Pk_cicz5, label='CIC Method (Corrected) z=5', marker='o', color='tomato')


    
    plt.scatter(k0_deconv, p0_deconv, alpha=0.5, label='CONCEPT snap z=5 deconv no fac', color='red',marker='_')
    plt.scatter(k1_deconv, p1_deconv, alpha=0.5, label='CONCEPT snap z=0 deconv no fac', color='orange',marker='_')
    plt.scatter(k_fftfac, Pk_fftfac, label='Class simulated Method z=0 with fac', marker='o', color='peru')
    plt.scatter(k_cicfac, Pk_cicfac, label='CIC Method (Corrected) z=0 with fac', marker='o', color='cyan')
    plt.scatter(k_fftz5fac, Pk_fftz5fac, label='Class simulated Method z=5 with fac', marker='o', color='greenyellow')
    plt.scatter(k_cicz5fac, Pk_cicz5fac, label='CIC Method (Corrected) z=5 with fac', marker='o', color='orchid')

    plt.plot(k_theory, Pk_theory, label='CLASS z = 0', linestyle='-', color='black')
    plt.plot(k_theory5, Pk_theory5, label='CLASS z = 5', linestyle='-', color='orange')
    plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k\ \left[1/\mathrm{Mpc}\right]$', fontsize=14)
    plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc})\right]$', fontsize=14)
    plt.xlim(1e-2, k_ny * 1.1)  
    plt.title(r'$P(k)$ created using CONCEPT HDF5 file CLASS vs CiC from CLASS', fontsize=16)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.show()
CiCvsSnap(snap_results=snap_results,plot=False, nodeconv=False, usefac=False, cicnofac=False)



#Loads the CONCEPT powerspectrum, which is saved in a text file (I refer to these as .00 files)
path = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Debugging og had\powerspec"

#Helper function to load the CONCEPT powerspectrum without all the unicode errors, #errors = 'ignore' is to ignore all unreadable bytes
def load_spectrum(filename):
    fullpath = os.path.join(path, filename)
    with open(fullpath, 'r', encoding='utf-8', errors='ignore') as f:
        k, P, P_corr = np.loadtxt(f, comments='#', usecols=(0,2,3), unpack=True)
    return k, P, P_corr

#k0, p0,p0_corr means a= 0.17 k1, p1, p1_corr means a = 1
k0, p0, p0_corr = load_spectrum('powerspec_a=0.17')
k1, p1, p1_corr = load_spectrum('powerspec_a=1.00')

#CONCEPTs own powerspectrum untouched
def conceptpowerspecplot(plot=False, showcorrected=False):
    if plot == False:
        return
    
    if showcorrected == True:
        plt.scatter(k1, p1_corr, label=r'CONCEPT $P(k)$ z=0 corrected', color='red')
        plt.scatter(k0, p0_corr, label=r'CONCEPT $P(k)$ z=5 corrected', color='yellow')
    plt.scatter(k1, p1, label=r'CONCEPT $P(k)$ z=0 raw', color='blue')
    plt.scatter(k0, p0, label=r'CONCEPT $P(k)$ z=5 raw', color='purple', )


    plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k\ \left[1/\mathrm{Mpc}\right]$', fontsize=14)
    plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc})\right]$', fontsize=14)
    plt.xlim(1e-2, k_ny * 2)  # eller fx 1e-2 til 5, hvis du hellere vil sætte fast grænse
    plt.title(r'Power Spectrum $P(k)$ exlusively from 00.files', fontsize=16)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.show()
conceptpowerspecplot(plot=False,showcorrected=False)


#plotting Class simulated CiC (with fac turned on) vs CONCEPTS .00 files--------------------------------
def CiCvsCONCEPT(plot = False,showcorrected=False,showCLASS = False):
    if plot== False:
        return
    
    if showcorrected == True:
        plt.scatter(k1, p1_corr, label=r'CONCEPT $P(k)$ z=0 corrected', color='red')
        plt.scatter(k0, p0_corr, label=r'CONCEPT $P(k)$ z=5 corrected', color='yellow')

    if showCLASS == True:
        plt.plot(k_theory, Pk_theory, label='CLASS z = 0', linestyle='-', color='black')
        plt.plot(k_theory5, Pk_theory5, label='CLASS z = 5', linestyle='-', color='orange')

    plt.scatter(k_fftfac, Pk_fftfac, label='Class simulated Method z=0 with fac', marker='o', color='peru')
    plt.scatter(k_cicfac, Pk_cicfac, label='CIC Method z=0 with fac', marker='o', color='cyan')
    plt.scatter(k_fftz5fac, Pk_fftz5fac, label='Class simulated Method z=5 with fac', marker='o', color='greenyellow')
    plt.scatter(k_cicz5fac, Pk_cicz5fac, label='CIC Method z=5 with fac', marker='o', color='orchid')

    plt.scatter(k1, p1, label=r'CONCEPT $P(k)$ z=0 raw', color='blue')
    plt.scatter(k0, p0, label=r'CONCEPT $P(k)$ z=5 raw', color='purple', )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k\ \left[1/\mathrm{Mpc}\right]$', fontsize=14)
    plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc})\right]$', fontsize=14)
    plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    plt.xlim(1e-2, k_ny * 1.1)  # eller fx 1e-2 til 5, hvis du hellere vil sætte fast grænse
    plt.title(r'Power Spectrum $P(k)$ exlusively from 00.files vs CIC', fontsize=16)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.show()
CiCvsCONCEPT(plot=False,showcorrected=False, showCLASS=True)

#Concept .00 files vs CONCEPT .hdf5 with CLASS as assistance--------------
def conceptpowerspecsnap(snap_results,plot=False, showcorrected=False, nodeconv=False):
    (
    k0_nodeconv, p0_nodeconv,
    k1_nodeconv, p1_nodeconv,
    k0_deconv,    p0_deconv,
    k1_deconv,    p1_deconv,
    k0_nodeconvfac, p0_nodeconvfac,
    k1_nodeconvfac, p1_nodeconvfac,
    k0_deconvfac,   p0_deconvfac,
    k1_deconvfac,   p1_deconvfac
    ) = snap_results

    if plot == False:
        return
    
    if showcorrected == True:
        plt.scatter(k1, p1_corr, label=r'CONCEPT $P(k)$ z=0 corrected', color='red')
        plt.scatter(k0, p0_corr, label=r'CONCEPT $P(k)$ z=5 corrected', color='yellow')
    
    
    plt.scatter(k1, p1, label=r'CONCEPT $P(k)$ z=0', color='blue')
    plt.scatter(k0, p0, label=r'CONCEPT $P(k)$ z=5', color='purple')

    if nodeconv == True:
        plt.scatter(k0_nodeconv, p0_nodeconv, alpha=0.5, label='CONCEPT snap z=5 no ', color='purple',marker='|')
        plt.scatter(k1_nodeconv, p1_nodeconv, alpha=0.5, label='CONCEPT snap z=0 no ', color='blue',marker='|')
    
    plt.scatter(k0_deconv, p0_deconv, alpha=0.5, label='CONCEPT snap z=5', color='red',marker='_')
    plt.scatter(k1_deconv, p1_deconv, alpha=0.5, label='CONCEPT snap z=0', color='orange',marker='_')
    plt.plot(k_theory, Pk_theory, label='CLASS z = 0', linestyle='-', color='black')
    plt.plot(k_theory5, Pk_theory5, label='CLASS z = 5', linestyle='-', color='orange')
    plt.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k\ \left[\mathrm{Mpc}^{-1}\right]$', fontsize=24)
    plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc}^3)\right]$', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(1e-2, k_ny * 1.1)  # eller fx 1e-2 til 5, hvis du hellere vil sætte fast grænse
    plt.title(r' Comparison of the $P(k)$ from .00 files vs $P(k)$ from .hdf5 files from CONCEPT', fontsize=24)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend(fontsize=20)
    plt.show()
conceptpowerspecsnap(snap_results, plot=True,showcorrected=False, nodeconv=False)
    
#Subplot structure instead:-----------------------------------------
def big_subplot(snap_results,plot=False):
    (
    k0_nodeconv, p0_nodeconv,
    k1_nodeconv, p1_nodeconv,
    k0_deconv,    p0_deconv,
    k1_deconv,    p1_deconv,
    k0_nodeconvfac, p0_nodeconvfac,
    k1_nodeconvfac, p1_nodeconvfac,
    k0_deconvfac,   p0_deconvfac,
    k1_deconvfac,   p1_deconvfac
    ) = snap_results

    if plot == False:
            return
    fig, axes = plt.subplots(6, 1, figsize=(18, 20),sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.5,right=0.7)

    # Subplot 1: CLASS only 
    ax = axes[0]
    ax.plot(k_theory, Pk_theory, label='CLASS z = 0', linestyle='-', color='black')
    ax.plot(k_theory5, Pk_theory5, label='CLASS z = 5', linestyle='-', color='orange')
    ax.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='both', ls='--', alpha=0.7)



    # Subplot 2 powerspectrum from CONCEPT only
    ax = axes[1]
    ax.scatter(k1, p1, label=r'CONCEPT $P(k)$ z=0 raw', color='blue')
    ax.scatter(k1, p1_corr, label=r'CONCEPT $P(k)$ z=0 corrected', color='red')
    ax.scatter(k0, p0, label=r'CONCEPT $P(k)$ z=5 raw', color='purple', )
    ax.scatter(k0, p0_corr, label=r'CONCEPT $P(k)$ z=5 corrected', color='yellow')
    ax.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='both', ls='--', alpha=0.7)


    # Subplot 3: CLASS + CONCEPT powerspectrums
    ax = axes[2]
    ax.plot(k_theory,  Pk_theory,  '-',  color='black',  label='CLASS z = 0')
    ax.plot(k_theory5, Pk_theory5, '-', color='orange', label='CLASS z = 5')
    ax.scatter(k1, p1, label=r'CONCEPT $P(k)$ z=0 raw', color='blue')
    ax.scatter(k1, p1_corr, label=r'CONCEPT $P(k)$ z=0 corrected', color='red')
    ax.scatter(k0, p0, label=r'CONCEPT $P(k)$ z=5 raw', color='purple')
    ax.scatter(k0, p0_corr, label=r'CONCEPT $P(k)$ z=5 corrected', color='yellow')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='both', ls='--', alpha=0.7)


    #Subplot 4, only snapshots
    ax = axes[3]
    ax.scatter(k0_nodeconv, p0_nodeconv, alpha=0.4, label='CONCEPT snap z=5 no deconv', color='purple',marker='|')
    ax.scatter(k0_deconv, p0_deconv, alpha=0.4, label='CONCEPT snap z=5 deconv', color='red',marker='_')
    ax.scatter(k1_nodeconv, p1_nodeconv, alpha=0.4, label='CONCEPT snap z=0 no deconv', color='blue',marker='|')
    ax.scatter(k1_deconv, p1_deconv, alpha=0.4, label='CONCEPT snap z=0 deconv', color='orange',marker='_')
    ax.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='both', ls='--', alpha=0.7)

    #subplot 5 Snapshots and CLASS
    ax = axes[4]
    ax.scatter(k0_nodeconv, p0_nodeconv, alpha=0.4, label='CONCEPT snap z=5 no deconv', color='purple',marker='|')
    ax.scatter(k0_deconv, p0_deconv, alpha=0.4, label='CONCEPT snap z=5 deconv', color='red',marker='_')
    ax.scatter(k1_nodeconv, p1_nodeconv, alpha=0.4, label='CONCEPT snap z=0 no deconv', color='blue',marker='|')
    ax.scatter(k1_deconv, p1_deconv, alpha=0.4, label='CONCEPT snap z=0 deconv', color='orange',marker='_')
    ax.plot(k_theory, Pk_theory, label='CLASS z = 0', linestyle='-', color='black')
    ax.plot(k_theory5, Pk_theory5, label='CLASS z = 5', linestyle='-', color='orange')
    ax.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='both', ls='--', alpha=0.7)


    # Subplot 6 all things in the same plot 
    ax = axes[5]
    ax.plot(k_theory,  Pk_theory,  '-',  color='black',  label='CLASS z = 0')
    ax.plot(k_theory5, Pk_theory5, '-', color='orange', label='CLASS z = 5')
    ax.scatter(k1, p1, label=r'CONCEPT $P(k)$ z=0 raw', color='blue')
    ax.scatter(k1, p1_corr, label=r'CONCEPT $P(k)$ z=0 corrected', color='red')
    ax.scatter(k0, p0, label=r'CONCEPT $P(k)$ z=5 raw', color='purple')
    ax.scatter(k0, p0_corr, label=r'CONCEPT $P(k)$ z=5 corrected', color='yellow')
    ax.scatter(k0_nodeconv, p0_nodeconv, alpha=0.4, label='CONCEPT snap z=5 no deconv', color='purple',marker='|')
    ax.scatter(k0_deconv, p0_deconv, alpha=0.4, label='CONCEPT snap z=5 deconv', color='red',marker='_')
    ax.scatter(k1_nodeconv, p1_nodeconv, alpha=0.4, label='CONCEPT snap z=0 no deconv', color='blue',marker='|')
    ax.scatter(k1_deconv, p1_deconv, alpha=0.4, label='CONCEPT snap z=0 deconv', color='orange',marker='_')
    ax.axvline(k_ny, color='gray',ls='--',label='Nyquist freq')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(which='both', ls='--', alpha=0.7)

    #Better legends + labels
    fig.supxlabel(r'$k\ [1/\mathrm{Mpc}]$', x=0.55, fontsize=14)
    fig.supylabel(r'$P(k)\ [\mathrm{Mpc}^3]$', fontsize=14)

    formatter = LogFormatterMathtext(labelOnlyBase=False)
    for ax in axes:
        ax.yaxis.set_major_formatter(formatter)

    # 
    # Hent alle handles og labels
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels  += l

    # Fjern dubletter, men behold rækkefølgen
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)

    # Tegn én samlet legend ude til højre
    fig.legend(
        uniq_h, uniq_l,
        loc='center left',
        bbox_to_anchor=(0.85, 0.5),   
        borderaxespad=0.0,           
        fontsize='small',
        frameon=True,
        fancybox=True,
        edgecolor='black'
    )

    plt.tight_layout(rect=(0,0,0.92,1))  
    plt.show()
big_subplot(snap_results,plot=False)