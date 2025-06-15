#!/usr/bin/env python3
from multiprocessing import Manager, Process, Pool
import queue
import pathlib
from pathlib import Path
import shutil
import subprocess
import os
import numpy as np
import re
import time
import random
import errno
import uuid
# Indstillinger----------------------------------------------------------- 
A_s_fid     = 2.105e-9          # Planck-værdi
sigma_A_s   = 0.030e-9        # 1-σ usikkerhed
n_s_value   = 0.9649            # Fast n_s
omega_cdm   = 0.12              # Fast Ω_cdm
N           = 32               #Mængden af partikler 
boxsize     = 500            #Box size i Mpc/h 
z_snapshot  = 0 #Redshift
# base_file   = "/home/justinfearis/concept/param/Bachelor/Til_NN/NN.param"   
base_file   = "/home/justinfearis/concept/param/Bachelor/Til_NN/NN.param"   
concept_bin = "/home/justinfearis/concept/concept"
output_dir = pathlib.Path("/home/justinfearis/concept/output/Til_rapport")
outdir      = pathlib.Path("Til_rapport")
outdir.mkdir(parents =True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
max_par     = 16       #Max antal kerner der kan tage 1 job
#parallel    = False     #For at parallelisere eller ej (For now så vil jeg gerne at den paralleliserer)(Aber das ist nicht working man)#Thomas siger jeg skal bruge xargs i stedet.


sigmaMult_test = 10
A_s_min       = (2.105  - 0.030  * sigmaMult_test) * 1e-9
A_s_max       = (2.105  + 0.030  * sigmaMult_test) * 1e-9

sigmaMult_train = 20
A_s_min_train       = (2.105  - 0.030  * sigmaMult_train) * 1e-9
A_s_max_train      = (2.105  + 0.030  * sigmaMult_train) * 1e-9

N_train = 180
N_val   = 20
N_test  = 100

np.random.seed(420)
A_s_train = np.random.uniform(A_s_min_train, A_s_max_train, size=N_train)

np.random.seed(7)
A_s_val   = np.random.uniform(A_s_min_train, A_s_max_train, size=N_val)

np.random.seed(39)
A_s_test  = np.random.uniform(A_s_min, A_s_max, size=N_test)

A_s_values = {
    "train": A_s_train,
    "val":   A_s_val,
    "test":  A_s_test,
}
F_seed = 42103431
np.random.seed(F_seed)
total_samples = len(A_s_train) + len(A_s_val) + len(A_s_test)
seed_amps   = np.random.randint(0, 2**31 - 1,size=total_samples)
seed_phases = np.random.randint(0, 2**31 - 1,size=total_samples)

splits   = {"train": 80, "val": 20, "test": 200}

#The initial worker, sets up all the global parameters for me to be able to use later)--------------------
def init_worker(job_q, train_list, val_list, test_list,lock, failed_list):
    global JOB_QUEUE, TRAIN_LIST, VAL_LIST, TEST_LIST, LOCK, FAILED_LIST
    JOB_QUEUE  = job_q
    TRAIN_LIST = train_list
    VAL_LIST   = val_list
    TEST_LIST  = test_list
    LOCK = lock
    FAILED_LIST = failed_list

#Laver parameterfiler (The main body):-----------------------------------------------------------------
def make_paramfile(A_s_value: float, split: str, idx:int, seed_amp: int, seed_phase:int) -> Path:
    """
    Returnerer sti til den parameter-fil CONCEPT skal køre og mappen den skal køre i.
    split = 'train' / 'val' / 'test'
    """
    A_s   = A_s_value
    tag   = f"id{idx:04d}_A_s{A_s:.20e}"
    sub   = "TrainingVal_Concept" if split in ("train", "val") else "Test_Data_concept"
    run_dir = output_dir / sub / tag


    dst = (outdir / f"params_{tag}.param").resolve()
    if dst.exists():
        dst.unlink()  # fjerner filen

    shutil.copy(base_file, dst)
    txt = dst.read_text().splitlines()
 

    
    # Updates 'A_s' in primordial_spectrum
    txt = [re.sub(r"^(\s*)'A_s'\s*:.*",
                  rf"\1'A_s': {A_s:.6e},", l) for l in txt]
    # fjern gamle global‑linjer -------------------------------------
    txt = [l for l in txt if not l.strip().startswith(
            ("parallel","job_directive","path.output_dir","path.job_dir",
            "random_seed", "random_seeds"))]

    txt = [
        f'path.output_dir = "{run_dir}"',
        f'path.job_dir    = "{run_dir}/job_{tag}"',
        'random_seeds = {',
         f"    'general'              : {0},",
         f"    'primordial amplitudes': {seed_amp},",
         f"    'primordial phases'    : {seed_phase},",
        '}',
        ""
    ] + txt
    # indsæt globale linjer EFTER blokken -------------------- AHHHHHHHHHHH

    dst.write_text("\n".join(txt))
    return dst

#Funktionen til at køre CONCEPT------------------------------------------------
def run_concept(cfg: Path, split: str, A_s: float):
    tag = cfg.stem.split('_', 1)[1]
    sub = "TrainingVal_Concept" if split in ("train","val") else "Test_Data_concept"
    run_dir = output_dir / sub / tag

    # Create run_dir once, with retry if parallel processes collide
    for _ in range(5):
        try:
            os.makedirs(run_dir, exist_ok=False)
            break
        except FileExistsError:
            time.sleep(random.uniform(0.01, 0.05))
    else:
        os.makedirs(run_dir, exist_ok=True)

    # Prepare job directory
    job_dir = run_dir / f"job_{tag}_{uuid.uuid4().hex[:6]}"
    job_dir.mkdir(parents=True, exist_ok=False)

    bin_dir = run_dir / "bin"
    bin_dir.mkdir(exist_ok=True)
    local_concept = bin_dir / "concept"
    shutil.copy(concept_bin, local_concept)
     # miljø
    env = os.environ.copy()
    env["OMP_NUM_THREADS"]    = "1"
    env["MKL_NUM_THREADS"]    = "1"
    fake_home = run_dir / "fake_HOME"
    fake_home.mkdir(exist_ok=True)
    env["HOME"]               = str(fake_home)
    # (your existing staging code here)

    cmd = [str(local_concept), "-p", str(cfg.resolve()), "-n", "1", "--local" ] 
    logpath = run_dir / f"{cfg.stem}.log"

    max_tries = 5
    for attempt in range(1, max_tries+1):
        with open(logpath, "a") as logfile:
            if attempt > 1:
                logfile.write(f"\n--- RETRY {attempt} efter staging- eller busy-fejl ---\n")
            try:
                # Run CONCEPT; do NOT re-mkdir run_dir here
                subprocess.check_call(
                    cmd,
                    cwd=str(job_dir),
                    env=env,
                    stdout=logfile,
                    stderr=subprocess.STDOUT
                )
                break  # succeeded
            except subprocess.CalledProcessError as e:
                err_txt = logpath.read_text()
                is_staging = "cannot create regular file" in err_txt
                if is_staging and attempt < max_tries:
                    time.sleep(random.uniform(0.01, 0.05))
                    continue
                raise RuntimeError(f"{tag} fejlede (exit {e.returncode}) – se {logpath}") from e
            except OSError as e:
                if e.errno == errno.ETXTBSY and attempt < max_tries:
                    logfile.write(f"OS error {e.errno} ({e.strerror}), retrying…\n")
                    time.sleep(random.uniform(0.01, 0.05))
                    continue
                raise

    # Clean up and return
    shutil.rmtree(fake_home, ignore_errors=True)
    return tag, split, A_s



def worker(_):
    while True:
        try:
            cfg, split, A_s = JOB_QUEUE.get_nowait()
        except queue.Empty:
            return

        # Every worker should (if it works) get one job and use run_concept
        try:
            tag, split_ret, A_s_ret = run_concept(cfg, split, A_s)
        except Exception as e:
            print(f"Job {cfg.stem} failed: {e}")
            with LOCK:
                FAILED_LIST.append((cfg, split, A_s, str(e)))
            continue

        # 
        subdir = "TrainingVal_Concept" if split in ("train", "val") else "Test_Data_concept"
        id_dir  = Path(output_dir) / subdir / tag
        start = time.time()
        snapshots = []
        while time.time() - start < 300:
            snapshots = list(id_dir.rglob("snapshots/*.h5")) + list(id_dir.rglob("snapshots/*.hdf5"))
            if snapshots:
                break
            time.sleep(2)

        if not snapshots:
            err = f"No snapshots found for {tag} after polling"
            print(f" {err}")
            with LOCK:
                FAILED_LIST.append((cfg, split, A_s, err))
            continue

        # Registers the results 
        entry = (A_s_ret, n_s_value, omega_cdm)
        with LOCK:
            if split_ret == "train":
                TRAIN_LIST.append(entry)
            elif split_ret == "val":
                VAL_LIST.append(entry)
            else:
                TEST_LIST.append(entry)

        print(f"{tag}: added to {split_ret}")
        return split, A_s
 


def write_vector(txtfile: Path, vec):
    """Skriv Cosmo-kompatibel tabel:
       header + (A_s, n_s, omega_cdm) for hvert run.
    """
    with open(txtfile, "w") as f:
        f.write("A_s \t n_s \t omega_cdm \n")          # header
        for A in vec:                                # one line pr sample 
            f.write(f"{A:.6e} \t {n_s_value:.6f} \t {omega_cdm:.6f} \n")
            
#Resume funktion, notice the missing parameters A_s_min and A_s_max-----------------------
def write_summaries(outdir: Path,
                    train_list: list[tuple[float,float,float]],
                    val_list:   list[tuple[float,float,float]],
                    test_list:  list[tuple[float,float,float]],
                    A_s_min:    float,
                    A_s_max:    float,
                    n_s_min:    float,
                    n_s_max:    float,
                    omega_cdm_min: float,
                    omega_cdm_max: float,
                    N_train:    int,
                    N_val:      int,
                    #N_test:     int,
                    z_snapshot: float,
                    boxsize:    float,
                    N:          int):
    # Witing out the box parameters for the neural network
    with open(outdir/"BoxParams.txt", "w") as f:
        f.write(f"{z_snapshot}\t{boxsize}\t{N}\n")

    # 2) MaxMinParams.txt
    with open(outdir/"MaxMinParams.txt", "w") as f:
        f.write(
            f"{A_s_min_train}\t{A_s_max_train}\t"
            f"{n_s_min}\t{n_s_max}\t"
            f"{omega_cdm_min}\t{omega_cdm_max}\t"
            f"{N_train + N_val}\n"
        )

    # 3) Hjælpefunktion til at skrive tre-kolonne‐filer
    def write_params_file(fname: str, entries: list[tuple[float,float,float]]):
        path = outdir/fname
        with open(path, "w") as f:
            f.write("A_s \t n_s \t omega_cdm\n")
            for A, n, o in entries:
                # Brug rå repræsentation uden formateringstricks
                f.write(f"{A}\t{n}\t{o}\n")

    # Writes out the three parameter files used 
    write_params_file("TrainingParams.txt", train_list)
    write_params_file("ValParams.txt",      val_list)
    write_params_file("TestParams.txt",     test_list)
#Rerun lost and dead runs------------------------------
MAX_RETRIES = 3    # how many times to try each job at most

def find_missing_jobs() -> list[tuple[Path,str,float]]:
    """
    Goes through all run‐dirs, finds those with no .hdf5 under snapshots/,
    and returns the corresponding (cfg, split, A_s) tuples to rerun.
    """
    missing = []
    for split, vec in [("train", A_s_train), ("val", A_s_val), ("test", A_s_test)]:
        for idx, A_s in enumerate(vec):
            tag = f"{split}_{idx:03d}_A_s{A_s:.2e}"
            sub = "TrainingVal_Concept" if split in ("train","val") else "Test_Data_concept"
            rundir = output_dir / sub / tag
            snap_glob = list(rundir.glob("**/snapshots/*.hdf5"))
            if not snap_glob:
                # reconstruct the param‐file path
                cfg = outdir / f"params_{tag}.param"
                missing.append((cfg, split, A_s))
    return missing
#Main-----------------------------------------------
MAX_RETRIES = 3

if __name__ == "__main__":
    mgr         = Manager()
    job_q       = mgr.Queue()
    train_list  = mgr.list()
    val_list    = mgr.list()
    test_list   = mgr.list()
    failed_list = mgr.list()
    lock        = mgr.Lock()

    idx = 0
    # Here the queue is filled with the jobs
    for split, vec in A_s_values.items():
        for A_s in vec:
            cfg = make_paramfile(A_s, split, idx, seed_amp=seed_amps[idx], seed_phase=seed_phases[idx])
            job_q.put((cfg, split, A_s))
            idx += 1

    #DE HER ER DEM SOM SKAL ÆNDRES FNJHISAOF
    parallel = False
    max_par = 4

    def run_pass(max_par):
        """Run exactly n_workers calls to worker() via Pool."""
        with Pool(
            processes=max_par,
            initializer=init_worker,
            initargs=(job_q, train_list, val_list, test_list, lock, failed_list)
        ) as pool:
            #her mapper jeg workers til mængden af jobs
            pool.map(worker, [None]*max_par)


    if parallel:
        print(f"Starting parallel pass with {max_par} workers …")
        run_pass(max_par)
    else:
        # manually bind the globals once
        init_worker(job_q, train_list, val_list, test_list, lock, failed_list)

        print("Running sequential pass …")
        # call worker() until queue is empty
        while not job_q.empty():
            worker(None)

    # Retry loop for failures only
    for attempt in range(1, MAX_RETRIES + 1):
        if not failed_list:
            print(f"All jobs succeeded after {attempt-1} retries.")
            break

        to_rerun = list(failed_list)
        failed_list[:] = []  # reset

        print(f"Retry {attempt}: re-running {len(to_rerun)} failed job(s).")
        # refill job_q with the failed ones
        for cfg, split, A_s, err in to_rerun:
            job_q.put((cfg, split, A_s))
        
        if parallel:
         run_pass(len(to_rerun))

        else:
                # sekventielt retry
                init_worker(job_q, train_list, val_list, test_list, lock, failed_list)
                while not job_q.empty():
                    worker(None)
    else:
            print(f"{len(failed_list)} jobs still failing after {MAX_RETRIES} retries:")
            for cfg, split, A_s, err in failed_list:
                print(f"  • {cfg.stem}: {err}")
    # Write summaries
    write_summaries(
        outdir        = outdir,
        train_list    = list(train_list),
        val_list      = list(val_list),
        test_list     = list(test_list),
        A_s_min       = A_s_min,
        A_s_max       = A_s_max,
        n_s_min       = n_s_value,
        n_s_max       = n_s_value,
        omega_cdm_min = omega_cdm,
        omega_cdm_max = omega_cdm,
        N_train       = N_train,
        N_val         = N_val,
        z_snapshot    = z_snapshot,
        boxsize       = boxsize,
        N             = N
    )

    # Move summary files to final output
    for fn in [
        "BoxParams.txt", "MaxMinParams.txt",
        "TrainingParams.txt", "ValParams.txt", "TestParams.txt"
    ]:
        os.replace(outdir / fn, output_dir / fn)































































#Alt under den her linje er gammelt og virker ikke længere-----------------------------------
# Seed_table = {"train": 42, "val": 7, "test": 39}
# if parallel:
#     print(f"Parallelt: kører op til {max_par} jobs ad gangen")
#     tags = []
#     with ThreadPoolExecutor(max_workers=max_par) as exe:
#         futures = [exe.submit(run_concept, cfg) for cfg in param_paths]
#         for fut in as_completed(futures):
#             tags.append(fut.result())
# else:
#     print(" Sekventiel mode: kører ét job")
#     tags = []
#     for cfg, split in param_paths:
#         tag = run_concept()
#         tags.appemd((tag,split))

# write_summaries(tags, output_dir)


# Parameter til NN -------------------------------------------------
# (outdir / "BoxParams.txt").write_text(f"{z_snapshot}\t{boxsize}\t{N}")
# write_vector(outdir / "TrainingParams.txt", train_params)
# write_vector(outdir / "ValParams.txt",      val_params)
# write_vector(outdir / "TestParams.txt",     test_params)

# all_As = np.concatenate([train_params, val_params, test_params])
# A_s_min, A_s_max = all_As.min(), all_As.max()
# (outdir / "MaxMinParams.txt").write_text(
#     f"{A_s_min}\t{A_s_max}\t{n_s_value}\t{n_s_value}\t"
#     f"{omega_cdm}\t{omega_cdm}\t{len(all_As)}"
# )

# # flytter de her txt filder til output_dir ---
# for fname in ["BoxParams.txt","MaxMinParams.txt",
#               "TrainingParams.txt","ValParams.txt","TestParams.txt"]:
#     shutil.move(outdir / fname, output_dir / fname)





# boxparams_str = f"{z_snapshot}\t{boxsize}\t{N}"
# (outdir / "BoxParams.txt").write_text(boxparams_str)


# write_vector(outdir/"TrainingParams.txt", train_params)
# write_vector(outdir/"ValParams.txt",      val_params)
# write_vector(outdir/"TestParams.txt",     test_params)

# all_As = np.concatenate([train_params, val_params, test_params])
# A_s_min, A_s_max = all_As.min(), all_As.max()
# maxmin_str = f"{A_s_min}\t{A_s_max}\t{n_s_value}\t{n_s_value}\t{omega_cdm}\t{omega_cdm}\t{len(all_As)}"
# (outdir / "MaxMinParams.txt").write_text(maxmin_str)

# for fname in ["BoxParams.txt","MaxMinParams.txt",
#               "TrainingParams.txt","ValParams.txt","TestParams.txt"]:
#     shutil.move(outdir/fname, output_dir/fname)

# print("✅ Alle param-filer genereret i", outdir)

# def main():
#     outdir.mkdir(exist_ok=True)
#     A_s_list = []
#     for delta_sigma in range(-20, 21):
#         cfg = make_paramfile(delta_sigma)
#         print(f"▶ Genereret og kører: {cfg.name}")
#         run_concept(cfg)
#         A_s_list.append(A_s_fid + delta_sigma * sigma_A_s)
# # Gem input-parametre
#     params_txt = outdir / 'TestParams_concept.txt'
#     with params_txt.open('w') as f:
#         f.write('A_s\tn_s\tomega_cdm\n')
#         for A_s in A_s_list:
#             f.write(f"{A_s:.6e}\t{n_s_value:.6f}\t{omega_cdm:.6f}\n")
#     print(f"Gemte input-parametre til {params_txt}")

# if __name__ == '__main__':
#     main()




# def is_file_locked(filepath):
#     if not os.path.exists(filepath):
#         return False
#     try:
#         os.rename(filepath, filepath)  # Prøver at "røre" ved filen
#         return False
#     except OSError:
#         return True
    
# if not is_file_locked(source_path):
#     shutil.copy2(source_path, dest_path)



# param_paths, train_params, val_params, test_params = [], [], [], []

# # ---- træning (i TrainingVal_Concept) ------------------------
# for idx, A_s in enumerate(A_s_train):
#     param = make_paramfile(A_s, "train", idx)      # split = 'train'
#     param_paths.append(param)
#     train_params.append(A_s)


# # ---- validering (samme mappe, andet seed) -------------------
# offset = len(A_s_train) 
# for idx, A_s in enumerate(A_s_val):
#     param = make_paramfile(A_s, "val",offset + idx)        # split = 'val'
#     param_paths.append(param)
#     val_params.append(A_s)

# offset += len(A_s_val) 
# # ---- test (egen mappe) --------------------------------------
# for idx, A_s in enumerate(A_s_test):
#     param = make_paramfile(A_s, "test", offset + idx)       # split = 'test'
#     param_paths.append(param)
#     test_params.append(A_s)

# for split, sigs in [("train", DELTA_SIGS_TRAIN),
#                     ("val",   DELTA_SIGS_VAL)]:
#     for ds in sigs:
#         param = make_paramfile(ds)
#         param_paths.append(param)
#         # run_concept(param)
#         (train_params if split=="train" else val_params).append(A_s_fid + ds*sigma_A_s)

# for ds in DELTA_SIGS_TEST:
#     param = make_paramfile(ds)
#     # snap = run_concept(param)
#     test_params.append(A_s_fid + ds*sigma_A_s)

#If parallel = True så burde det her køre:-------------------------------
# def write_vector(txtfile: Path, vec):
#     """Som i Cosmo-scriptet: én værdi pr. linje."""
#     with open(txtfile, "w") as f:
#         for v in vec:
#             f.write(f"{v:.6e}\n")







# def make_paramfile(A_s_value: float, split: str, idx: int) -> Path:
#     """
#     Returnerer sti til CONCEPT-parameterfil + opretter run-mappe.
#     split = 'train' / 'val' / 'test'
#     """
#     # 1) Initial opsætning
#     A_s   = A_s_value
#     tag   = f"id{idx:04d}_A_s{A_s:.20e}"
#     sub   = "TrainingVal_Concept" if split in ("train","val") else "Test_Data_concept"
#     run_dir = output_dir / sub / tag
#     run_dir.mkdir(parents=True, exist_ok=True)

#     # 2) Copy base_file til dst
#     dst = (outdir / f"params_{tag}.param").resolve()
#     if dst.exists():
#         dst.unlink()
#     shutil.copy(base_file, dst)

#     # 3) Læs og rens bort gamle cluster-indstillinger
#     lines = dst.read_text().splitlines()
#     lines = [
#         l for l in lines
#         if not l.strip().startswith((
#             "parallel", "job_directive",
#             "path.output_dir", "path.job_dir",
#             "random_seed", "random_seeds"
#         ))
#     ]

#     # 4) Indsæt dine globale overrides øverst
#     seed = Seed_table[split]
#     header = [
#         "parallel = False",
#         f'path.output_dir = "{run_dir}"',
#         f'path.job_dir    = "{run_dir}/job_{tag}"',
#         "random_seeds = {",
#         f"    'general'              : {seed},",
#         f"    'primordial amplitudes': {seed},",
#         f"    'primordial phases'    : {seed},",
#         "}",
#         ""
#     ]

#     # 5) Opdater værdien af A_s inde i primordial_spectrum-blokken
#     #    (gøres efter header, så din A_s-linje ikke ryddes væk!)
#     def update_A_s(line):
#         return re.sub(
#             r"^(\s*)'A_s'\s*:.*",
#             rf"\1'A_s': {A_s:.6e},",
#             line
#         )

#     new_lines = []
#     for l in lines:
#         if "'A_s'" in l:
#             new_lines.append(update_A_s(l))
#         else:
#             new_lines.append(l)

#     # 6) Skriv alt tilbage til dst
#     dst.write_text("\n".join(header + new_lines))
#     return dst



# #Til træning og validering
# def write_params(txtfile, values):
#     with open(txtfile, "w") as f:
#         for v in values:
#             f.write(f"{v:.5e}\n")


#Funktionen til at lave parameterfilerne for hver +-sigmaA_S 
# def make_paramfile(delta_sigma: int) -> pathlib.Path:
#     A_s = A_s_fid + delta_sigma * sigma_A_s
#     tag = f"{delta_sigma:+05.1f}sigma"
#     dst = outdir / f"params_{tag}.param"
#     shutil.copy(base_file, dst)
#     lines = dst.read_text().splitlines()
#     new_lines = []
#     in_block = False
#     for line in lines:
#         if line.strip().startswith("primordial_spectrum"):
#             in_block = True
#             new_lines.append(line)
#             continue
#         # Tjek om vi lige afsluttede blokken
#         if in_block and line.strip().startswith("}"):
#             in_block = False
#             new_lines.append(line)
#             continue
#         # Hvis vi er inde i blokken, og rammer 'A_s', så erstat
#         if in_block and "'A_s'" in line:
#             indent = line[:len(line) - len(line.lstrip())]
#             new_lines.append(f"{indent}'A_s': {A_s:.6e},")
#         else:
#             new_lines.append(line)
#     dst.write_text("\n".join(new_lines))
#     return dst

# def run_concept(cfg: Path, split:str, A_s: float) -> tuple[str,str,float]:
#     tag = cfg.stem.split('_', 1)[1]         # "±005sigma"
#     log = cfg.with_suffix(".log")
#     run_dir = output_dir / ("TrainingVal_Concept" if split in ("train","val")
#                             else "Test_Data_concept") / tag
    
#     job_dir  = run_dir / f"job_{tag}"                        # én mappe pr. run
#     job_dir.mkdir(parents=True, exist_ok=True)
#     local_concept = run_dir / "concept"
#     shutil.copy(concept_bin, local_concept)

#     staging = run_dir / "concept" / "job"
#     if staging.exists():
#         shutil.rmtree(staging)
#     staging.mkdir(parents=True, exist_ok=True)

#     cmd = [
#         str(local_concept),
#         "-p", str(cfg.resolve()),
#         "-n", "1",               # kører på 1 kerne for at undgå load imbalances
#         "--local",
#         #"-j",    "--exclusive",
#     ]
#     env = os.environ.copy()
#     env["OMP_NUM_THREADS"] = "1"
#     env["MKL_NUM_THREADS"] = "1"
#     with open(log, "w") as logfile:
#         try:
#             subprocess.check_call(cmd,
#                                   env=env,
#                                   cwd=str(run_dir),
#                                    stdout=logfile, 
#                                    stderr=subprocess.STDOUT)
#         except subprocess.CalledProcessError as e:
#             # e.returncode er exit-koden fra CONCEPT
#             raise RuntimeError(f"{tag} fejlede (exit {e.returncode}) - se {log}") from e
    
#     return tag, split, A_s


# def write_summaries(results: list[tuple[str,str]], outdir: Path):
#     """
#     Skriver BoxParams.txt, MaxMinParams.txt og {Training,Val,Test}Params.txt én gang.
#     """
#     # BoxParams.txt
#     (outdir / "BoxParams.txt").write_text(
#         f"{z_snapshot} \t {boxsize} \t {N}\n"
#     )

#     # MaxMinParams.txt
#     (outdir / "MaxMinParams.txt").write_text(
#         f"{A_s_min:.6e} \t {A_s_max:.6e} \t "
#         f"{n_s_value:.6f} \t {n_s_value:.6f} \t "
#         f"{omega_cdm:.6f} \t {omega_cdm:.6f} \t "
#         f"{(N_val + N_train)}\n"
#     )
#     # Training/Val/TestParams.txt – med header og tre kolonner
#     for split in ("train", "val", "test"):
#         fname = f"{split.capitalize()}Params.txt"
#         split_tags = [tag for tag, s in results if s == split]
#         with open(outdir / fname, "w") as f:
#             # Header identisk med Cosmo-scriptet
#             f.write("A_s \t n_s \t omega_cdm \n")
#             for tag in split_tags:
#                 A = float(tag.replace("A_s", ""))
#                 # Vi antager her, at n_s og omega_cdm er konstante for alle runs:
#                 f.write(f"{A} \t {n_s_value} \t {omega_cdm} \n")



    # mgr      = Manager()
    # job_q    = mgr.Queue()
    # train_list = mgr.list()
    # val_list   = mgr.list()
    # test_list  = mgr.list()
    # lock     = mgr.Lock()
    # for cfg, split, A_s in jobs:
    #     job_q.put((cfg, split, A_s))
    # # 3) spawn processes
    # procs = [Process(target=worker, args=(job_q,train_list, val_list,test_list ,lock)) for _ in range(max_par)]
    # for p in procs: p.start()
    # for p in procs: p.join()
    # write_summaries(
    #     outdir         = outdir,           # Path til din midlertidige mappe
    #     train_list     = train_list,       # List of (A_s, n_s, omega_cdm) for træning
    #     val_list       = val_list,         # List of (A_s, n_s, omega_cdm) for validering
    #     test_list      = test_list,        # List of (A_s, n_s, omega_cdm) for test
    #     A_s_min        = A_s_min,          # global min A_s
    #     A_s_max        = A_s_max,          # global max A_s
    #     n_s_min        = n_s_value,        # global min n_s (her konstant)
    #     n_s_max        = n_s_value,        # global max n_s (her konstant)
    #     omega_cdm_min  = omega_cdm,        # global min Ω_cdm (her konstant)
    #     omega_cdm_max  = omega_cdm,        # global max Ω_cdm (her konstant)
    #     N_train        = N_train,          # antal træningsprøver
    #     N_val          = N_val,            # antal valideringsprøver
    #     #N_test         = N_test,           # antal testprøver (kan fjernes hvis ubrugt)
    #     z_snapshot     = z_snapshot,       # redshift
    #     boxsize        = boxsize,          # box‐side
    #     N              = N                 # antal partikler
    # )
    # for fn in ["BoxParams.txt","MaxMinParams.txt","TrainingParams.txt","ValParams.txt","TestParams.txt"]:
    #     os.replace(outdir/fn, output_dir/fn)





# def run_concept(cfg: Path, split: str, A_s: float) -> tuple[str,str,float]:
#     tag     = cfg.stem.split('_',1)[1]
#     run_dir = output_dir / (
#         "TrainingVal_Concept" if split in ("train","val")
#         else "Test_Data_concept"
#     ) / tag
#     run_dir.mkdir(parents=True, exist_ok=True)

#     job_dir = run_dir / f"job_{tag}"
#     job_dir.mkdir(parents=True, exist_ok=True)

#     # lille pause for at sikre ingen kollision mellem jobs
#     time.sleep(random.uniform(0.005, 0.02))
#     unique = uuid.uuid4().hex[:8]
#     job_dir = run_dir / f"job_{tag}_{unique}"
#     # Prøv at lave mappen — failer, hvis der allerede findes
#     job_dir.mkdir(parents=True, exist_ok=False)

   

#     # kopier binær


#     cmd = [str(local_concept), "-p", str(cfg.resolve()), "-n", "1", "--local"]
#     logpath = run_dir / f"{cfg.stem}.log"

#     # ── RETRY-LOOP ───────────────────────────────────────────
#     max_tries = 5
#     for attempt in range(1, max_tries+1):
#         with open(logpath, "a") as logfile:
#             if attempt > 1:
#                 logfile.write(f"\n--- RETRY {attempt} efter staging- eller busy-fejl ---\n")
#             try:
#                 # Prøv at starte CONCEPT
#                 os.makedirs(run_dir, exist_ok=False)
#                 subprocess.check_call(
#                     cmd,
#                     cwd=str(bin_dir),
#                     env=env,
#                     stdout=logfile,
#                     stderr=subprocess.STDOUT
#                 )
#                 break  # succeeded
#             except subprocess.CalledProcessError as e:
#                 err_txt = logpath.read_text()
#                 is_staging = "cannot create regular file" in err_txt
#                 if is_staging and attempt < max_tries:
#                     time.sleep(random.uniform(0.01, 0.05))
#                     continue
#                 raise RuntimeError(f"{tag} fejlede (exit {e.returncode}) – se {logpath}") from e
#             except OSError as e:
#                 # Text file busy?
#                 if e.errno == errno.ETXTBSY and attempt < max_tries:
#                     logfile.write(f"OS error {e.errno} ({e.strerror}), retrying…\n")
#                     time.sleep(random.uniform(0.01, 0.05))
#                     continue
#                 # andre OS-fejl skal videre
#                 raise
#             else:
#                  os.makedirs(run_dir, exist_ok=True)
#     subprocess.run(cmd, check=True)
#     # clean up
#     return tag, split, A_s