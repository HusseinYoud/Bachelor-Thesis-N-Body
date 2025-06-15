"""Imports & setting up træningdata"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*layer.add_variable.*")          #Tror ikke umiddelbart de her warnings er noget der kan gøres noget ved
warnings.filterwarnings("ignore", message=".*RandomNormal is unseeded.*")         #Tror ikke umiddelbart de her warnings er noget der kan gøres noget ved, måske der kan ved den her?
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                   #Fjerner en notice om noget numerisk precision or smth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'                    #Fjerner notice om at CPU bliver brugt til at optimize stuff, kan måske fjerne relevante ting også not sure so be careful
import numpy as np
import matplotlib.pyplot as plt
import time
from classy import Class
import psutil
import gc
from joblib import Parallel, delayed
from scipy.interpolate import interp1d #Chatten siger det her skal bruges til at interpolere P(k) værdierne til den der 128/128 grid
plt.rc("axes", labelsize=30, titlesize=32)   # skriftstørrelse af xlabel, ylabel og title
plt.rc("xtick", labelsize=26, top=True, direction="in")  # skriftstørrelse af ticks, vis også ticks øverst og vend ticks indad
plt.rc("ytick", labelsize=26, right=True, direction="in") # samme som ovenstående
plt.rc("legend", fontsize=30) # skriftstørrelse af figurers legends
plt.rcParams["font.size"] = "20"
plt.rcParams["figure.figsize"] = (16,9)


def createR(N, dim = 3, mu = 0, sig = 1/np.sqrt(2)):
    """Funktion til at lave R
    args:
    ----------
    N : int
        Sidelængde for kassen
    """
    size = [N] * dim
    c_1 = np.random.normal(mu, sig, size = size)
    c_2 = np.random.normal(mu, sig, size = size)

    R = c_1 + 1j * c_2

    """Enforcer hermitisk symmetri"""
    R[0,0,0] = R[0,0,0].real     
    for i in np.arange(N):
        for j in np.arange(N):
            for k in np.arange(N):
                # R[i,j,k] = np.conj(R[N - i - 1, N - j - 1, N - k - 1])      #forstår ummidelbart ikke hvorfor den her line ikke virker, burde gøre det samme som de to næste samlet no?
                ii, jj, kk = -i % N, -j % N, -k % N
                # if (i,j,k) != (ii,jj,kk):                                     #Ikke sikker på det her overhovedet sparer tid. hvertfald ikke meget hvis det gør
                R[ii, jj, kk] = np.conj(R[i, j, k])
   
    return R

def power_spectrum(A_s = 2.1e-9, n_s = 0.9649, omega_b = 0.02237, omega_cdm = 0.12, z = 5, k = np.logspace(-4, 0, 500), plot = False, returnk = False):
    """In ChatGPT we trust"""
    # Set cosmological parameters (ΛCDM model with Planck 2018 values)
    params = {
        'output': 'mPk',          # Output matter power spectrum
        'H0': 67.36,              # Hubble parameter [km/s/Mpc]
        'omega_b': omega_b,       # Baryon density
        'omega_cdm': omega_cdm,        # Cold Dark Matter density
        'Omega_k': 0.0,           # Spatial curvature (flat universe)
        'n_s': n_s,            # Scalar spectral index
        'A_s': A_s,            # Primordial curvature power
        'z_pk': z,              # Redshift for power spectrum calculation
        'non linear': 'none',     # Linear power spectrum (set to True for nonlinear)
        'z_max_pk': 1000            #gør ingenting me thinks, vist bare max for z_pk
    }

    #Chatten siger man skal gøre sådan her idk?
    # k_sim = k_mag.flatten() 
    # k_min_sim = k_sim.min()
    # k_max_sim = k_sim.max()
    # params['P_k_max_1/Mpc'] = k_max_sim * 1.1

    # Initialize CLASS and compute the cosmology
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()


    #Til den mærkelige 128/128


    #Gør k 1D
    k = k.flatten()

    # Calculate P(k) at z 
    Pk = np.array([cosmo.pk(ki, params['z_pk']) for ki in k])
    # Pk = np.array([cosmo.pk(ki, params['z_pk']) if k > 0 else 0 for ki in k])
    # kth_sim = np.logspace(np.log10(k_min_sim), np.log10(k_max_sim), 500)  # 500 punkter fra k_min→k_max
    # Pk_sim = np.array([cosmo.pk(kk, params['z_pk']) for kk in kth_sim])
    # interp  = interp1d(kth_sim, Pk_sim, bounds_error=False, fill_value=0.0)
    # Pk      = interp(k_sim).reshape(k_mag.shape)

    # Plotting
    if plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(k, Pk, lw=2, color='navy')

        plt.xlabel(r'$k\ \left[h/\mathrm{Mpc}\right]$', fontsize=14)
        plt.ylabel(r'$P(k)\ \left[(\mathrm{Mpc}/h)^3\right]$', fontsize=14)
        plt.title(f"Linear Matter Power Spectrum at z = {params['z_pk']}", fontsize=16)
        plt.grid(True, which='both', ls='--', alpha=0.7)
        plt.xlim(1e-4, 1e0)
        plt.tight_layout()

    # Clean up CLASS instance
    # cosmo.struct_cleanup()
    cosmo.struct_cleanup()
    cosmo.empty()
    del cosmo

    if returnk == True:
        return k, Pk#, Hz, D_z
    else:
        return Pk

def find_delta(z, L, N, A_s = 2.1e-9, n_s = 0.9649, omega_b = 0.02237, omega_cdm = 0.12):
    """Find delta using power_spectrum and create_R
    args:
    -------------------
    z : int
        redshift at which to find delta
    
    L : int
        Size of box in Mpc
    
    N : int
        Grids points on each axis
    """

    k, pk = power_spectrum(A_s = A_s, n_s = n_s, omega_b = omega_b, omega_cdm = omega_cdm, z = z, k = k_mag, plot = False, returnk = True)
    R = createR(N)

    v = L**3 
    fac= np.sqrt(2 * np.pi / v) * N**3    #delta_k = sqrt(p_k), vil gerne have det i et NxNxN grid though så skal reshape også
    delta_k = np.sqrt(pk.reshape(k_mag.shape)) * fac                             
    delta = np.fft.ifftn(delta_k * R).real
    
    del delta_k, R
    return delta.astype(np.float64)

def makeFolders(savepath):
    """Laver folders til at gemme data i"""
    files = ('', 'Training & val data', 'Test data')

    for file in files:
        if not os.path.exists(savepath + file):
            os.mkdir(savepath + file)
            print(f"Folder '{savepath + file}' created.")
        else:
            print(f"Folder '{savepath + file}' already exists.")

def saveDelta(i,z, L, N, savepath, A_s = 2.1e-9, n_s = 0.9649, omega_cdm = 0.12, N_samples = 5000, overWrite = False, omega_b = 0.02237, TrainingNoise = False):
    # cumtime = 0
    if not overWrite:
        if os.path.exists(savepath + f'Training & val data/delta_train_id-{i + 1}.npy'):                      #Hvis filen allerede eksisterer så skip til næste, gør at der kan laves store sæt over flere gange
            return                                                                                            #Bare for at stoppe den, returner none though men vidst ikke en bedre måde at gøre det på
    # temptime = time.time()

    delta = find_delta(z, L, N, A_s = A_s, n_s = n_s, omega_b = omega_b, omega_cdm = omega_cdm)
    if TrainingNoise:                                                                                   #Mulighed for at tilføje noise til træningssættet, "if" lader ikke til at slow det down (like 0.5 sec max for 10 mil loops)
        delta += np.random.normal(0, 0.1 * np.max(delta), (N, N, N))
    np.save(savepath + f'Training & val data/delta_train_id-{i + 1}', delta)                          #Gemmer dem med unik ID for hver fil, burde gøre at man kan gemme så mange der er plads til på hardisken i stedet for at være limited af ram :)
    

    # del delta
    # gc.collect()
    # if i % 10 == 0 and i != 0:                                                                      #Bare for at kunne se progress på dem der tager lidt længere tid
    #     cumtime += (time.time() - temptime) * 10                                                        #Kan godt flyttes udenfor for at blive mere præcis i guess (og så ikke gange med 10) men er bare et estimat anyways
    #     print(f'Progress: {np.round((i / N_samples) * 100, 4)} % Expected remaining time: {round(cumtime/i * (N_samples - i), 2)} s')


def createData(N_samples, z, L, N, ValSize = 0.2, A_s_min = 2.1e-9, A_s_max = 2.1e-9, n_s_min = 0.9649, n_s_max = 0.9649, omega_cdm_min = 0.12, omega_cdm_max = 0.12, omega_b = 0.02237, TestData = False, TrainingNoise = False, savepath = None, SameTrain = False, paramRepeat = 1, overWrite = False):
    """Function to create data for training and sampling
    """
    """Sætter op de potentiele variable, hopefully foolproof til så at kunne implementere de andre senere, ved ikke om det måske kunne laves pænere med params dictionary eller noget - kig på det hvis tid"""
    if A_s_max != A_s_min:
        np.random.seed(420) ; A_s_train = np.random.uniform(A_s_min, A_s_max, size = int(N_samples * (1 - ValSize)))                     #Sætter random seed for at kunne fortsætte kode
        np.random.seed(7) ; A_s_val = np.random.uniform(A_s_min, A_s_max, size = int(N_samples * ValSize))                              #sætter nyt random seed for at validation data ikke bare svarer til den første del af træningsdataet (og 7 er det mest tilfældige tal)
    else:
        A_s_train = [A_s_max] * int(N_samples * (1 - ValSize))
        A_s_val = [A_s_max] * int(N_samples * ValSize)

    if n_s_max != n_s_min:
        np.random.seed(420) ; n_s_train = np.random.uniform(n_s_min, n_s_max, size = int(N_samples * (1 - ValSize)))
        np.random.seed(7) ; n_s_val = np.random.uniform(n_s_min, n_s_max, size = int(N_samples * ValSize))
    else:
        n_s_train = [n_s_max] * int(N_samples * (1 - ValSize))
        n_s_val = [n_s_max] * int(N_samples * ValSize)

    if omega_cdm_max != omega_cdm_min:
        np.random.seed(420) ; omega_cdm_train = np.random.uniform(omega_cdm_min, omega_cdm_max, size = int(N_samples * (1 - ValSize)))
        np.random.seed(7) ; omega_cdm_val = np.random.uniform(omega_cdm_min, omega_cdm_max, size = int(N_samples * ValSize))
    else:
        omega_cdm_train = [omega_cdm_max] * int(N_samples * (1 - ValSize))
        omega_cdm_val = [omega_cdm_max] * int(N_samples * ValSize)


                                       #Har unik A,n,omega for hver delta (fastholder ikke nogen af dem )
    

    A_s_train = np.repeat(A_s_train, paramRepeat)               ; A_s_val = np.repeat(A_s_val, paramRepeat)             #Hvis man vil lavere flere deltasæt for hver sæt af parametre. np.repeat er noget hurtigere end manuel list comprehension
    n_s_train = np.repeat(n_s_train, paramRepeat)               ; n_s_val = np.repeat(n_s_val, paramRepeat)
    omega_cdm_train = np.repeat(omega_cdm_train, paramRepeat)   ; omega_cdm_val = np.repeat(omega_cdm_val, paramRepeat)


    if savepath is not None and TestData == False:                                                          #Gemmer parametrene brugt til at lave træningssæt, gemmes bare i en array da det ikke fylder særlig meget alligevel og det gør det nemmere at sortere ting senere
        with open(savepath + 'TrainingParams.txt', mode = 'w') as f:                                        #Kunne godt bruge np.save men havde allerede lavet det her da jeg fandt ud af at den eksisterede so here we are
            f.write('A_s \t n_s \t omega_cdm \n') 
            for nrA, A in enumerate(A_s_train):
                f.write(f'{A} \t {n_s_train[nrA]} \t {omega_cdm_train[nrA]} \n')   


        with open(savepath + 'ValParams.txt', mode = 'w') as f:                                             #Samme som ovenfor bare med validation parametre
            f.write('A_s \t n_s \t omega_cdm \n') 
            for nrA, A in enumerate(A_s_val):
                f.write(f'{A} \t {n_s_val[nrA]} \t {omega_cdm_val[nrA]} \n')   

    np.random.seed()                                                                                                    #Resetter random seed for at R-felterne er random

    Parallel(n_jobs=-1, verbose = 10)(delayed(saveDelta)(i, z, L, N, savepath, A_s_train[i], n_s_train[i], omega_cdm_train[i], N_samples, overWrite)            #Parallelisering af at lave data, tager for some reason arguments i en seperat parantes, n_jobs = -1 betyder brug alle cores
        for i in range(len(A_s_train)))                                                                                                              #https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html
                                                                                                                                                     #Lader til at batch_size = 1 er det den oftest lander på, kunne overveje at fastsætte den der for at den ikke springer så meget når man resumer
    Parallel(n_jobs=-1, verbose = 10)(delayed(saveDelta)(i + len(A_s_train), z, L, N, savepath, A_s_val[i], n_s_val[i], omega_cdm_val[i], N_samples, overWrite) #same men testdata
        for i in range(len(A_s_val)))

    """Parallel er omkring ~1.7 gange hurtigere (706 kontra 417 sek). For 4 cores er bedst teoretisk speedup apparently 2.44x (Amdahl's law) hvis 90% af koden er perfekt paralleliseret -hvilket den ikke er da der også er overhead & stuff
    Er desuden noget pickling (???????????????????? er det navn xd) der tager noget tid.
    https://chatgpt.com/c/68091c21-cfcc-8009-b661-41f9ff506d7e   for nogle tips der måske kan speede det lidt mere op - at udkommentere manuel del delta og gc lader til at speede lidt up
    Måske batch sizes men standard er auto så tror det er fine? 
    Dask måske bedre end joblib for store opgaver
    """

def createTestData(N_samples, z, L, N, A_s_min = 2.1e-9, A_s_max = 2.1e-9, n_s_min = 0.9649, n_s_max = 0.9649, omega_cdm_min = 0.12, omega_cdm_max = 0.12, omega_b = 0.02237, TestData = False, TrainingNoise = False, savepath = None, SameTrain = False):
    """Function to create data for training and sampling
    """
    """Sætter op de potentiele variable, hopefully foolproof til så at kunne implementere de andre senere, ved ikke om det måske kunne laves pænere med params dictionary eller noget - kig på det hvis tid"""
    np.random.seed(39)                          #Vil ikke have at testdata har samme seed som træningsdata
    if A_s_max != A_s_min:
        A_s_train = np.random.uniform(A_s_min, A_s_max, size = N_samples)
    else:
        A_s_train = [A_s_max] * N_samples

    if n_s_max != n_s_min:
        n_s_train = np.random.uniform(n_s_min, n_s_max, size = N_samples)
    else:
        n_s_train = [n_s_max] * N_samples

    if omega_cdm_max != omega_cdm_min:
        omega_cdm_train = np.random.uniform(omega_cdm_min, omega_cdm_max, size = N_samples)
    else:
        omega_cdm_train = [omega_cdm_max] * N_samples

    np.random.seed()                                                                            #Resetter random seed for at R-felterne er random

    cumtime = 0
    for nrA, (A, n, o) in enumerate(zip(A_s_train, n_s_train, omega_cdm_train)):
        temptime = time.time()
        delta = find_delta(z, L, N, A_s = A, n_s = n, omega_b = omega_b, omega_cdm = o)
        if TrainingNoise:                                                                       #Mulighed for at tilføje noise til træningssættet, "if" lader ikke til at slow det down (like 0.5 sec max for 10 mil loops)
            delta += np.random.normal(0, 0.1 * np.max(delta), (N, N))

        np.save(savepath + f'Test data/delta_test_id-{nrA + 1}', delta)                         #Gemmer dem med unik ID for hver fil, burde gøre at man kan gemme så mange der er plads til på hardisken i stedet for at være limited af ram :)

        if nrA % 10 == 0 and nrA != 0:                                                          #Bare for at kunne se progress på dem der tager lidt længere tid
            cumtime += (time.time() - temptime) * 10                                            #Kan godt flyttes udenfor for at blive mere præcis i guess (og så ikke gange med 10) men er bare et estimat anyways
            print(f'Progress: {np.round((nrA / N_samples) * 100, 4)} % Expected remaining time: {round(cumtime/nrA * (N_samples - nrA), 2)} s')
            # mem = psutil.virtual_memory()
            # print(f"Memory usage: {mem.percent}% used")

    with open(savepath + 'TestParams.txt', mode = 'w') as f:                                             #Samme som ovenfor bare med validation parametre
        f.write('A_s \t n_s \t omega_cdm \n') 
        for nrA, A in enumerate(A_s_train):
            f.write(f'{A} \t {n_s_train[nrA]} \t {omega_cdm_train[nrA]} \n')   


z = 5
L = 10000
N = 32
N_samples = 10
N_samples_test = 20

sigmaMult = 5
A_s_min = (2.105 - 0.030 * sigmaMult) * 1e-9                #https://arxiv.org/pdf/1807.06209#page=19&zoom=100,56,722 side 16 omega_cdm bare den de kalder omega_c i assume
A_s_max = (2.105 + 0.030 * sigmaMult) * 1e-9                #Prøv måske at implementer joblib parallel (tror den hedder delayed den der skal bruges) processing

sigmaMult_test = 5
A_s_min       = (2.105  - 0.030  * sigmaMult_test) * 1e-9
A_s_max       = (2.105  + 0.030  * sigmaMult_test) * 1e-9

sigmaMult_train = 10
A_s_min_train       = (2.105  - 0.030  * sigmaMult_train) * 1e-9
A_s_max_train      = (2.105  + 0.030  * sigmaMult_train) * 1e-9
n_s_min = (0.9665 - 0.0038 * sigmaMult)                     
n_s_max = (0.9665 + 0.0038 * sigmaMult)
omega_cdm_min = (0.11933 - 0.00091 * sigmaMult)              
omega_cdm_max = (0.11933 + 0.00091 * sigmaMult)

#For CONCEPT anbefaler chatGPT at holde A_s indenfor ~1-5, n_s ~0.9-1.1 og omega_cdm ~(0.1-0.5)h^2
#RESULTATER lader umiddelbart til at være bedre hvis den får lov at træne på +-10 sigma (kan måske endda gå større?) også selvom testdata kun er +-5 sigma

# A_s_min = 2.1e-9 ; A_s_max = 2.1e-9
n_s_min = 0.9649 ; n_s_max = 0.9649 
omega_cdm_min = 0.12 ; omega_cdm_max = 0.12

"""Bare for at undgå at køre dem inde i funktionen mange gange"""
k_vals = 2 * np.pi * np.fft.fftfreq(N, d= L/N)                              #d = afstand mellem punkter, N = antal punkter
kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing = 'ij')           #Lav det til et grid
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)                                      #Størrelsen i hvert punkt af vores grid
k_mag[0,0,0] = 1e-10


savepath = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Varians test\\"
makeFolders(savepath = savepath)                                    #Laver folders til at gemme data i       

with open(savepath + 'BoxParams.txt', mode = 'w') as f:             #Gemmer values brugt for boks og parametre
    f.write(f'{z} \t {L} \t {N}')
with open(savepath + 'MaxMinParams.txt', mode = 'w') as f:
    f.write(f'{A_s_min} \t {A_s_max} \t {n_s_min} \t {n_s_max} \t {omega_cdm_min} \t {omega_cdm_max} \t {N_samples}')


createTestData(N_samples_test, z, L, N, A_s_min = A_s_min, A_s_max = A_s_max, n_s_min = n_s_min, n_s_max = n_s_max, omega_cdm_min = omega_cdm_min, omega_cdm_max = omega_cdm_max, savepath = savepath, TestData = True)

createData(N_samples, z, L, N, 
                                A_s_min = A_s_min_train, A_s_max = A_s_max_train,
                                n_s_min = n_s_min, n_s_max = n_s_max,
                                omega_cdm_min = omega_cdm_min, omega_cdm_max = omega_cdm_max, 
                                savepath = savepath, SameTrain = False, paramRepeat = 1, overWrite = True, ValSize=0.2)            #Hurtig test med paramRepeat = 5 lader ikke til at være meget forskellig fra 1, om noget slightly værre (total mængde samples 5k i begge tilfælde)




"""
spørgsmål:
Bliver det ikke ineffektivt at generere data når man får mange parametre? Skal vel have et loop for hver parameter - behøver måske ikke, prøv bare at vary alle på samme tid som beskrevet - skal dog nok stadig have mange samples

fix save tingen da createdata er blevet lavet mere simpel, evt bare fjern sametrain


"""
