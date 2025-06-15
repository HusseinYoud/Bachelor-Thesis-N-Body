import os
import shutil
import argparse
import importlib.util
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
import numpy as np
import os, types, re
import copy
from matplotlib import rcParams
from matplotlib.collections import PathCollection, LineCollection
import tensorflow as tf
from tf_keras import mixed_precision,Model
#Lige nu har vi de her eksperimenter som vi vil køre
Data_dir = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Fodringspipeline\Datafolder4"
# ---------- Configuration Defaults ----------
OVERRIDE_EPOCHS = 1000
OVERRIDE_BATCH_SIZE = 16
OVERRIDE_LR = 4e-3
# External scripts
DATA_CREATE_SCRIPT = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Fodringspipeline\Cosmo data create (new).py"
FRAP_SCRIPT         = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Fodringspipeline\FraP_kTilDelta.py"
COSMO_SCRIPT        = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Fodringspipeline\Cosmo NN5.py"

tf.config.optimizer.set_jit(True)          # XLA-fusion på alle grafer
mixed_precision.set_global_policy('float32')   # Udnyt Tensor Cores (GPU)

script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import FraP_kTilDelta as frap   # loads the file normally, forces an inclusion of `import numpy as np`
import numpy as _np

_original_compile = Model.compile          # gem originalen

def _compile_with_jit(self, *args, **kwargs):
    kwargs.setdefault("jit_compile", True)  # tilføj, hvis ikke sat
    return _original_compile(self, *args, **kwargs)

Model.compile = _compile_with_jit           # patch globalt
def import_cosmo_module(custom_data_dir):
    """
    Indlæser “Cosmo NN5.py” som et modul, men uden at eksekvere dens top‐level trænings‐block.
    I stedet overskriver vi:
      1) path, model_savepath, figpath og data_savepath → peger i stedet på custom_data_dir
      2) Indlæser BoxParams.txt + MaxMinParams.txt fra custom_data_dir
      3) Beregner variabler og data, så vi får y_train, y_val, y_test, partition, labels osv.
    Til sidst returnerer vi modulet, hvor alle funktioner (get_data, DataGenerator, make_model osv.) er tilgængelige.
    """
    # 1) Læs Cosmo NN5.py som tekstlinjer
    with open(COSMO_SCRIPT, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # 2) Byg ny “path = ”-linje ud fra custom_data_dir
    dd = custom_data_dir
    if not dd.endswith(os.sep):
        dd += os.sep
    escaped = dd.replace("\\", "\\\\")
    new_path_line = f'path = "{escaped}"'

    patched = []
    replaced_path = False
    skip_mode = False

    for raw in lines:
        stripped = raw.lstrip()

        # A) Erstat den første forekomst af “path =”
        if (not replaced_path) and stripped.startswith("path ="):
            patched.append(new_path_line)
            replaced_path = True
            continue

        # B) Find “B = test_og_plot_models(…” og drop ALLE linjer indtil “plt.show()”
        if stripped.startswith("B = test_og_plot_models"):
            skip_mode = True
            continue

        if skip_mode:
            # Fortsæt med at droppe linjer, indtil vi rammer “plt.show”
            if stripped.startswith("plt.show"):
                skip_mode = False
            continue

        # C) Uden for skip_mode: behold linjen uændret
        patched.append(raw)

    patched_source = "\n".join(patched)

    # 3) Opret et tomt modul og eksekver det patched-script i det namespace
    spec = importlib.util.spec_from_loader("cosmo_nn5_module", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cosmo_nn5_module"] = module
    exec(compile(patched_source, COSMO_SCRIPT, "exec"), module.__dict__)

    # 4) Nu skal vi overskrive diverse globaller i modulet, så det peger på custom_data_dir
    module.path = custom_data_dir if custom_data_dir.endswith(os.sep) else custom_data_dir + os.sep
    # Vælg hvor vi gemmer weights og figurer, inden for samme custom_data_dir, fx:
    module.model_savepath = os.path.join(custom_data_dir, "checkpoint.weights.h5")
    module.figpath        = os.path.join(custom_data_dir, "figurer")  # fx en undermappe til plots
    module.data_savepath  = os.path.join(custom_data_dir, "plotdata") # fx en undermappe til save‐data

    # Opret mapperne, hvis de ikke findes:
    os.makedirs(module.figpath, exist_ok=True)
    os.makedirs(module.data_savepath, exist_ok=True)

    # 5) Læs BoxParams.txt og MaxMinParams.txt fra custom_data_dir – præcis som top‐level i Cosmo NN5
    box_file    = os.path.join(module.path, "BoxParams.txt")
    maxmin_file = os.path.join(module.path, "MaxMinParams.txt")

    module.z, module.L, module.N = np.loadtxt(box_file, dtype=int)
    (
        module.A_s_min, module.A_s_max,
        module.n_s_min, module.n_s_max,
        module.omega_cdm_min, module.omega_cdm_max,
        _unused_Nsamples
    ) = np.loadtxt(maxmin_file)

    # 6) Beregn hvilke parametre der varierer, og antallet af dem
    module.vary_flags, module.vary_number = module.get_variable_params(
        module.A_s_min, module.A_s_max,
        module.n_s_min, module.n_s_max,
        module.omega_cdm_min, module.omega_cdm_max
    )

    # 7) Hent y_train, y_val, y_test ved at kalde get_data(path)
    #    (get_data bruger altid “TrainingParams.txt”, “ValParams.txt” og “TestParams.txt” i module.path)
    module.y_train, module.y_val, module.y_test = module.get_data(module.path)

    # 8) Lav partition og labels, præcis som top‐level:
    module.partition, module.labels = module.make_partition_and_labels(
        N_samples=len(module.y_train) + len(module.y_val),
        y_train=module.y_train,
        y_val=module.y_val
    )

    return module

EXPERIMENTS = {
  "CLASS2CONCEPT": {
    "train_dir":   "Training & val data",
    "val_dir":     "Training & val data",
    "test_dir":    "Test data_CONCEPT",
    "params_file": "TrainingParams.txt",      #  
    "val_params":  "ValParams.txt",           # 
    "test_params": "TestParams_CONCEPT.txt"   #  
  },
    "CONCEPT3": {
        "train_dir": "Training & val data_CONCEPT",
        "val_dir":   "Training & val data_CONCEPT",
        "test_dir":  "Test data_CONCEPT",
        "params_file": "TrainingParams_CONCEPT.txt",      
        "val_params":  "ValParams_CONCEPT.txt",           
        "test_params": "TestParams_CONCEPT.txt"   
  },
}


def run_frap_to_delta(data_dir, ngrid):
    # Define roots and outputs exactly as before...
    test_root     = os.path.join(data_dir, 'Test_Data_concept')
    trainval_root = os.path.join(data_dir, 'TrainingVal_Concept')

    test_output     = os.path.join(data_dir, 'Test data_CONCEPT')
    trainval_output = os.path.join(data_dir, 'Training & val data_CONCEPT')
    os.makedirs(test_output, exist_ok=True)
    os.makedirs(trainval_output, exist_ok=True)

    # Now the calls:
    test_files = frap.find_hdf5_files(test_root)
    for idx, h5file in enumerate(sorted(test_files), start=1):
        pos   = frap.read_positions(h5file)
        box   = frap.read_boxsize(h5file)
        dens  = frap.cic_density(pos, box, ngrid)    # np is already in frap’s globals
        delta = frap.compute_delta(dens).astype(_np.float64)
        delta = frap.deconvolve_cic(delta, box, ngrid).astype(_np.float64)
        #delta -= delta.mean()
        out_path = os.path.join(test_output, f"delta_test_id-{idx}.npy")
        _np.save(out_path, delta)
        test_files = frap.find_hdf5_files(test_root)

    trainval_files = frap.find_hdf5_files(trainval_root)
    for idx, h5file in enumerate(sorted(trainval_files), start=1):
        pos   = frap.read_positions(h5file)
        box   = frap.read_boxsize(h5file)
        dens  = frap.cic_density(pos, box, ngrid)    # np is already in frap’s globals
        delta = frap.compute_delta(dens).astype(_np.float64)
        delta = frap.deconvolve_cic(delta, box, ngrid).astype(_np.float64)
        delta -= delta.mean()
        out_path = os.path.join(trainval_output, f"delta_train_id-{idx}.npy")
        _np.save(out_path, delta)


# -------- 1) Data Creation --------
def import_module(name, path):
    import importlib.util, sys, os
    # Remove any cached module
    if name in sys.modules:
        del sys.modules[name]
    # Ensure the script's directory is on sys.path so local imports (e.g. tf_keras) work
    module_dir = os.path.dirname(path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    # Create a fresh module spec
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    # If module needs common globals, inject them here
    # Now execute module code
    spec.loader.exec_module(mod)
    return mod

def rmtree_force(path):
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=lambda f, p, e: os.chmod(p, 0o777) or f(p))




# sørg for, at stien til Cosmo_NN5.py er i sys.path, så import_module virker


# -------- 1) Data Creation --------

BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def run_data_creation(z, L, N,
                      n_trainval, n_test, base_dir,
                      *,
                      A_s_min_train=None, A_s_max_train=None,
                      A_s_min_test =None, A_s_max_test =None):
    """
    Opret CLASS-felter.  Hvis brugeren ikke giver et interval,
    falder vi tilbage på et ±σ-bånd omkring 2.105×10⁻⁹.
    """

    # ---------- 1) fastlæg grænser ----------
    if A_s_min_train is None or A_s_max_train is None:
        sigma_tv       = 20
        A_s_min_train  = (2.105 - 0.030*sigma_tv) * 1e-9
        A_s_max_train  = (2.105 + 0.030*sigma_tv) * 1e-9
    if A_s_min_test is None or A_s_max_test is None:
        sigma_test     = 10
        A_s_min_test   = (2.105 - 0.030*sigma_test) * 1e-9
        A_s_max_test   = (2.105 + 0.030*sigma_test) * 1e-9

    # ---------- 2) lav mapper ----------
    save_base = base_dir if base_dir.endswith(os.sep) else base_dir + os.sep
    os.makedirs(os.path.join(save_base, 'Test data'),          exist_ok=True)
    os.makedirs(os.path.join(save_base, 'Training & val data'), exist_ok=True)

    # ---------- 3) importer generator-modulet ----------
    data_mod = import_module('datacreator', DATA_CREATE_SCRIPT)

    # ---------- 4) generér datasæt ----------
    data_mod.createTestData(
        N_samples=n_test,   z=z, L=L, N=N, savepath=save_base,
        A_s_min=A_s_min_test,  A_s_max=A_s_max_test,
        TestData=True
    )
    data_mod.createData(
        N_samples=n_trainval, z=z, L=L, N=N, savepath=save_base,
        A_s_min=A_s_min_train, A_s_max=A_s_max_train,
        ValSize=0.1, overWrite=True
    )
    return save_base



def run_experiment(name, cfg, data_dir):
    print(f"\n--- Running experiment: {name} ---")

    params_file_abspath = os.path.join(data_dir, cfg["params_file"])
    val_params_abspath  = os.path.join(data_dir, cfg["val_params"])
    test_params_abspath = os.path.join(data_dir, cfg["test_params"])
    train_dir_abspath   = os.path.join(data_dir, cfg["train_dir"])
    val_dir_abspath     = os.path.join(data_dir, cfg["val_dir"])
    test_dir_abspath    = os.path.join(data_dir, cfg["test_dir"])
    print(f"  → Trænings-mappe     : {train_dir_abspath}")
    print(f"  → Validerings-mappe : {val_dir_abspath}")
    print(f"  → Test-mappe        : {test_dir_abspath}")

    # Check eksistens…
    for f in (params_file_abspath, val_params_abspath, test_params_abspath):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Parameterfil '{f}' mangler.")
    for d in (train_dir_abspath, val_dir_abspath, test_dir_abspath):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Datamappe '{d}' mangler.")

    # Importér patched Cosmo NN5
    cosmo = import_cosmo_module(data_dir)

    # Overskriv hvor DataGenerator leder efter delta_train_*.npy osv.
    cosmo.train_dir = train_dir_abspath + os.sep
    cosmo.val_dir   = val_dir_abspath   + os.sep
    cosmo.test_dir  = test_dir_abspath  + os.sep

    # 1) Brug partition['train'] direkte
    train_IDs = cosmo.partition['train']
    val_IDs   = cosmo.partition['validation']

    training_generator = cosmo.DataGenerator(
        list_IDs   = train_IDs,
        labels     = cosmo.labels,
        batch_size = OVERRIDE_BATCH_SIZE,
        dim        = (cosmo.N, cosmo.N, cosmo.N),
        n_channels = 1,
        shuffle    = True,
        augmentation= True
    )
    validation_generator = cosmo.DataGenerator(
        list_IDs   = val_IDs,
        labels     = cosmo.labels,
        batch_size = OVERRIDE_BATCH_SIZE,
        dim        = (cosmo.N, cosmo.N, cosmo.N),
        n_channels = 1,
        shuffle    = False,
        augmentation= False
    )
    

    # Kald test_og_plot_models (håndterer test-IDs internt)
    uncertainty_method = "MC"
    print(f"→ Sender til test_og_plot_models: epochs={OVERRIDE_EPOCHS}, batch_size={OVERRIDE_BATCH_SIZE}, lr={OVERRIDE_LR}, uncertaintyEst={uncertainty_method}")
    results = cosmo.test_og_plot_models(
        training_generator,
        validation_generator,
        testsamples     = len(cosmo.y_test),
        epochs          = OVERRIDE_EPOCHS,
        LearningRate    = OVERRIDE_LR,
        uncertaintyEst  = uncertainty_method
    )
    fig = plt.gcf()
    ax_loss, ax_As, ax_ns, ax_om = fig.axes        # row-major

    # ---------- 3) tynd ud i scatter-plottet ----------
    STEP = 10                                      # ← vis hver 10. punkt
    barcol = next((c for c in ax_As.collections
               if isinstance(c, LineCollection)), None)

    if barcol is not None:
        segs = np.array(barcol.get_segments())      # (N, 2, 2)
        xs   = segs[:, 0, 0]
        y_lo = segs[:, 0, 1]
        y_hi = segs[:, 1, 1]
        ys   = 0.5 * (y_lo + y_hi)

        # to-sidet fejl: [[lower], [upper]]
        yerr = np.vstack((ys - y_lo, y_hi - ys))

        keep = np.arange(0, len(xs), STEP, dtype=int)

        # fjern ALLE gamle punkt- og fejlkunster
        for art in list(ax_As.lines) + list(ax_As.collections):
            art.remove()

        # 1-til-1-linje
        ax_As.plot([xs.min(), xs.max()],
                [xs.min(), xs.max()],
                lw=1, ls="--", color="b", zorder=1)

        # nye, nedprøvede punkter + fejlbjælker
        ax_As.errorbar(xs[keep], ys[keep], yerr=yerr[:, keep],
                    fmt="o", ms=5,
                    ecolor="cyan", elinewidth=1.2, capsize=2,
                    markerfacecolor="cyan", markeredgecolor="k",
                    lw=0, zorder=2)

    # ---------- 4) layout-tweaks ----------
    ax_loss.set_position([0.06, 0.18, 0.36, 0.72])   # bredde 0.36 → luftspalte
    ax_As  .set_position([0.58, 0.18, 0.36, 0.72])

    if name.upper() == "CONCEPT3":
        ax_loss.set_yscale("log")
        ax_loss.set_ylim(1e-6, 1e0)
        ax_loss.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=2))
        ax_loss.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0e"))

    # ---------- 5) ryd nederste akser ----------
    ax_ns.remove();  ax_om.remove()

    # ---------- 6) styling + gem ----------
    plt.rcParams.update({"font.size": 16})
    fig.set_size_inches(11.7, 4.2)
    fig.suptitle(
        f"{name}: {OVERRIDE_EPOCHS} epochs,  LR={OVERRIDE_LR:.1e},  "
        f"{len(cosmo.y_train)} train / {len(cosmo.y_val)} val",
        y=1.07, fontsize=18
    )
    out_png = os.path.join(data_dir, f"{name}_results.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"→ Eksperiment '{name}' afsluttet (figur gemt som {out_png})")
    return results


# ------------------- Main Pipeline -------------------
def main_pipeline():
    # Hard-coded parameters
    z          = 0
    L          = 500
    N          = 32
    n_trainval = 200
    n_test     = 100
    ngrid      = N
    data_dir   = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Fodringspipeline\Datafolder4"
    
    data_dir = os.path.join(os.path.dirname(__file__), "Datafolder4")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"'Datafolder4' mangler i {os.path.dirname(__file__)}")

    # Run steps
    run_data_creation(z, L, N, n_trainval, n_test,
                    base_dir=data_dir)
    run_frap_to_delta   (data_dir, ngrid)
    dats_dir = os.path.join(BASE_DIR, "Datafolder4")
    for name, cfg in EXPERIMENTS.items():
        run_experiment(name, cfg, data_dir)
    print("Alle eksperimenter gennemført.")

    print("Pipeline done. Results in", data_dir)

# Kick off automatically
if __name__ == '__main__':
    main_pipeline()
