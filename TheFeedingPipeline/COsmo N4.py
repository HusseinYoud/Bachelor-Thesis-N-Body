"""Imports & setting up træningdata"""
import numpy as np
import os
import psutil
import sys
print(f"[CosmoN4] pid={os.getpid()} argv={sys.argv!r}")


# #Opsætter multithreading 
os.environ["OMP_NUM_THREADS"]           = "16"   # Antal tråde til OpenMP
#os.environ["KMP_BLOCKTIME"]            = "30"  # hvor længe en tråd venter før den går i dvale, pisse irrelevant her
os.environ["KMP_AFFINITY"]             = "granularity=fine,compact,8,0"
os.environ["KMP_SETTINGS"]             = "0"
p = psutil.Process()  
p.cpu_affinity([0, 1, 2, 3])

import warnings
warnings.filterwarnings("ignore", message=".*layer.add_variable.*")          #Tror ikke umiddelbart de her warnings er noget der kan gøres noget ved
warnings.filterwarnings("ignore", message=".*RandomNormal is unseeded.*")         #Tror ikke umiddelbart de her warnings er noget der kan gøres noget ved, måske der kan ved den her?
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                   #Fjerner en notice om noget numerisk precision or smth, gør det til gengæld også like 100% langsommere & might just kill everything so ye, dont do this
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                    #Fjerner notice om at CPU bliver brugt til at optimize stuff, kan måske fjerne relevante ting også not sure so be careful
import tensorflow as tf
import tensorflow_probability as tfp
from tf_keras import layers, models, optimizers, callbacks, utils, backend
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
import matplotlib.pyplot as plt
from pathlib import Path
import time

import argparse

# ----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500,
                    help="Antal trænings-epochs (standard 500)")
args = parser.parse_args()
# ----------------------------------------------------



import random
plt.rc("axes", labelsize=30, titlesize=32)   # skriftstørrelse af xlabel, ylabel og title
plt.rc("xtick", labelsize=26, top=True, direction="in")  # skriftstørrelse af ticks, vis også ticks øverst og vend ticks indad
plt.rc("ytick", labelsize=26, right=True, direction="in") # samme som ovenstående
plt.rc("legend", fontsize=30) # skriftstørrelse af figurers legends
plt.rcParams["font.size"] = "20"
plt.rcParams["figure.figsize"] = (16,9)

BASE_DIR = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN"



def loss_mse(y_true, y_pred):
    """Standard mean squared error loss function for Tensorflow to use & abuse"""
    y_pred = tf.squeeze(y_pred)  # Removes singleton dimensions → shape (batch_size,)
    y_true = tf.squeeze(y_true)  # Den anbefaler at bruge dem for robustness, kan også specify [: , 0] på dem begge eller bare undlade det

    # print(f'{y_pred = }')     # for deubbigng
    # print(f'{y_true = }')

    loss = (y_pred - y_true)**2         #Tror bare den tager det her for hver individuel komponent
    return tf.reduce_mean(loss)  #tf.reduce_mean minder om (måske nøjagtig samme?) som np.mean()  - https://stackoverflow.com/questions/34236252/what-is-the-difference-between-np-mean-and-tf-reduce-mean

def loss_nlll(y_true, y_pred, eps=1e-6):
    """Gaussian negative log likelihood loss function for Tensorflow to use & abuse"""
    mu = y_pred[:, 0]
    sig_raw = y_pred[:, 1]

    sigma = tf.math.softplus(sig_raw) + eps  # Ensures sigma is positive
    var = sigma**2

    mu = tf.squeeze(mu)
    var = tf.squeeze(var)
    y_true = tf.squeeze(y_true)       #Håbede det her måske kunne fikse det men lader til allerede at have den rigtige shape :/

    loss = 0.5 * (tf.math.log(var) + (y_true - mu)**2 / var) #+ tf.abs(tf.math.log(var)) * 0.01   #sidste term er band-aid, lader til at virke er bare fishy om sigma så rent faktisk kan tolkes som noget relevant, svarer til L1 regularization måske?
    return tf.reduce_mean(loss)

def loss_nlll(y_true, y_pred, eps=1e-6):
    """Negative log-likelihood for 3 parameters with mu and sigma each."""
    mu_A = y_pred[:, 0]
    log_sig_A = y_pred[:, 1]
    mu_n = y_pred[:, 2]
    log_sig_n = y_pred[:, 3]
    mu_o = y_pred[:, 4]
    log_sig_o = y_pred[:, 5]

    # Ensure positive sigma with softplus
    sig_A = tf.math.softplus(log_sig_A) + eps
    sig_n = tf.math.softplus(log_sig_n) + eps
    sig_o = tf.math.softplus(log_sig_o) + eps

    # Compute NLL for each parameter
    loss_A = 0.5 * (tf.math.log(sig_A**2) + (y_true[:, 0] - mu_A)**2 / sig_A**2)
    loss_n = 0.5 * (tf.math.log(sig_n**2) + (y_true[:, 1] - mu_n)**2 / sig_n**2)
    loss_o = 0.5 * (tf.math.log(sig_o**2) + (y_true[:, 2] - mu_o)**2 / sig_o**2)

    return tf.reduce_mean(loss_A + loss_n + loss_o)

def chi_squared_calc(y_expected, y_measured, y_err):
    chi_squared = 0
    for i in range(len(y_expected)):
        chi_squared += (y_expected[i] - y_measured[i])**2 / y_err[i]**2

    return chi_squared

def sigma_coverage_calc(y_expected, y_measured, y_err, sigmaMult):
    N = len(y_expected)
    coverage = 0
    for i in range(N):
        if abs(y_expected[i] - y_measured[i]) < (y_err[i] * sigmaMult):
            coverage += 1/N
    return coverage

def abs_percent_deviation_calc(y_expected, y_measured):     #abs percentage error nok mere apt navn
    abs_deviation = abs(y_expected - y_measured) / y_expected
    dev_mu = np.mean(abs_deviation) * 100
    dev_std = np.std(abs_deviation, ddof = 1) * 100
    return np.array([dev_mu, dev_std])

def ting_til_prior(n):     #For at undgå errors når man loader weights da der ellers bruges en unnamed lambda funktion
    def _prior(t):
        return tfd.Independent(
            tfd.Normal(loc=tf.zeros(n), scale=1.0),
            reinterpreted_batch_ndims=1
        )
    return _prior

def prior(kernel_size, bias_size=0, dtype=None):                     #https://kzhu.ai/tensorflow-probabilistic-deep-learning-models/
    """Prior distribution for Bayesian layers."""
    n = kernel_size + bias_size                                      #Total antal parametre
    return models.Sequential([tfp.layers.DistributionLambda(ting_til_prior(n = n))])

def posterior(kernel_size, bias_size=0, dtype=None):
    """Posterior distribution for Bayesian layers (trainable)."""
    n = kernel_size + bias_size                                     #Total antal parametre
    return models.Sequential([
        tfp.layers.VariableLayer(                                   #Variable layer gør det trainable afaik
            tfp.layers.IndependentNormal.params_size(n),            #Normalfordeling for hver 
            dtype=dtype
        ),
        tfp.layers.IndependentNormal(n, convert_to_tensor_fn=tfd.Distribution.sample),
    ])

def get_variable_params(A_s_min, A_s_max, n_s_min, n_s_max, omega_cdm_min, omega_cdm_max):
    """Finder which parameters vary and how many there are"""
    vary_flags = [A_s_max != A_s_min, n_s_max != n_s_min, omega_cdm_max != omega_cdm_min]                                   #liste med True/False så den ved hvilke parametre der varierer og skal bruges
    vary_numbah = 0
    for flag in vary_flags:
        if flag:
            vary_numbah += 1 
    return vary_flags, vary_numbah

def combine_y_data(arrays, flags):
    return np.column_stack([arr for arr, f in zip(arrays, flags) if f])                     #bruger np.column stack for at få det på (N_samples, #varierende parametre) form som det skal bruges som senere

def get_data(path, Normalize = True):
    """Aint pretty but it works, actually it is disguting please help
    Returnerer bare trænings, validation og test data og parametre"""
    A_train, n_train, o_train = np.loadtxt(path + 'TrainingParams.txt', skiprows = 1).T     #får value unpack error whatever yada yada uden .T
    A_val, n_val, o_val = np.loadtxt(path + 'ValParams.txt', skiprows = 1).T
    A_test, n_test, o_test = np.loadtxt(path + 'TestParams.txt', skiprows = 1).T

    if Normalize:
        A_train /= A_s_max ; A_val /= A_s_max ; A_test /= A_s_max

        n_train /= n_s_max ; n_val /= n_s_max ; n_test /= n_s_max

        o_train /= omega_cdm_max ; o_val /= omega_cdm_max ; o_test /= omega_cdm_max


    train_params = [A_train, n_train, o_train]
    val_params = [A_val, n_val, o_val]
    # test_params = [A_test, n_test, o_test]
    
    y_train = combine_y_data(train_params, vary_flags)                                         #gør så at y-labels kun indeholder varierende parametre
    y_val   = combine_y_data(val_params,   vary_flags)
    # y_test  = combine_y_data(test_params,  vary_flags)
    y_test = np.array([*zip(A_test, n_test, o_test)])                                          #laver bare test på den gamle måde, slipper for at ændre plotning & stuff dernede så, påvirker ikke modellen either way da det bare er til plot

    return y_train, y_val, y_test




def random_flip_3d(volume):
    """Er ikke sikker på hvor glad jeg er for potentielt kun at flippe en eller to akser, burde være 0 eller alle no?
    -nvm, giver rigtig fin mening
    
    """
    # Flip along each axis with 50% chance
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=0)
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=1)
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=2)
    return volume

def random_rotate90_3d(volume):
    # Choose one of the three planes and rotate 0, 90, 180, or 270 degrees
    k = np.random.randint(0, 4)
    # k = 4
    plane = random.choice([(0,1), (1,2), (0,2)])
    return np.rot90(volume, k=k, axes=plane)




def make_partition_and_labels(N_samples, y_train, y_val, valSize = 0.2):
    """Laver partition & labels for træning og validation for at kunne holde styr på træningsdata
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    partition = {'train': [], 'validation': [] }
    labels = {}

    TrainSize = len(y_train) #int(N_samples * (1 - valSize))      #Kunne vel egentlig bare bruge len(y_train) now that i think about it
    ValSize = len(y_val) #int(N_samples * valSize)

    for i in range(TrainSize):
        partition['train'].append(f'id-{i + 1}')
        labels[f'id-{i + 1}'] = y_train[i]

    for i in range(ValSize):
        partition['validation'].append(f'id-{TrainSize + i + 1}')
        labels[f'id-{i + 1 + TrainSize}'] = y_val[i]

    return partition, labels
    
class DataGenerator(utils.Sequence):
    'Generates data for Keras - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(16,16,16), n_channels=1,
                  shuffle=True, augmentation = False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]       #sætter vores index til et subset af alle index, vil nok selv have skrevet det som [index * batch_size : index * batch_size + batch_size], vælger altså bare index 
                                                                                        #fra et random index og frem (note at det stadig er random though da on_epoch_end shuffler det i starten og efter hver epoch)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            Xi = np.load(path + 'Training & val data//' + 'delta_train_' + ID + '.npy')
            Xi = Xi[..., np.newaxis]
            if self.augmentation:
                Xi = random_flip_3d(Xi)
                Xi = random_rotate90_3d(Xi)
            X[i,] = Xi

            y.append(self.labels[ID])
        y = np.array(y)
        return X, y
 
def make_model(augmentation = False, convLayers = 1, convSize = None, convDepth = None, GAP = True, denseLayers = 1, denseWidth = None, DropoutRate = 0.2, fullDR = False, uncertaintyEst = None):
    """Function to generate a tensorflow model
    args:
    ------------------
    normalisering : bool
        Whether there should be a normalizing layer or not

    augmentation : bool
        Whether there should be a data augmentation layer (here rotation & random flips) or not

    convLayers : int
        Amount of convolutional layers

    convSize : array/list?
        Sizes of the convolutional layers, leave unspecificed or make sure its length is convLayers
    
    convDepth : array/list?
        Number of filters in each convolutional layer

    GAP : bool
        Whether to use GlobalAveragePooling or not (uses flatten if False)

    denseLayers : int:
        Amount of dense layers (excluding final output layer)

    DropoutRate : float
        Rate used for dropout layers

    fullDR : bool
        Whether to only apply DR on the last layer or every layer

    uncertaintyEst : string
        Method used to estimate the uncertainty on predictions, must be either 'nlll', 'MC' or 'Bayesian'
    """
    if uncertaintyEst == None:
        raise ValueError("uncertaintyEst must be either 'nlll', 'MC' or 'Bayesian'")

    if convSize == None:
        convSize = [(3,3,3)] * convLayers
    elif len(convSize) != convLayers:                           #Check at alle conv layers har en tilhørende size, ellers giv error
        raise ValueError('convSize must have length convLayers')
        
    if convDepth == None:                                       #Hvis ikke specified convDepth så bare lav standard for dem alle
        convDepth = [32] * convLayers
    elif len(convDepth) != convLayers:                           #Check at alle conv layers har en tilhørende depth, ellers giv error
        raise ValueError('convDepth must have length convLayers')
    
    if denseWidth == None:
        denseWidth = [64] * denseLayers
    elif len(denseWidth) != denseLayers:
        raise ValueError('denseWidth must have length denseLayers')

    
    model = models.Sequential([
        layers.Input(shape = (N,N,N)),                          #Giver input size, det vil den for some reason hellere have for sig ind i reshape
        layers.Reshape((N, N, N, 1))])
    
    # if normalisering:                                           #Med eller uden normalisering
    #     normalizer = layers.Normalization(axis = None)            #Giver problemer når vi arbejder med batches da normalisering mellem batches ikke nødvendigvis er ens, specielt for små batches
    #     normalizer.adapt(X_train)
    #     model.add(normalizer)

    #if augmentation:

    for nr, size in enumerate(convSize):                                       #Adder convolutional layers
        model.add(layers.Conv3D(convDepth[nr], size, activation = 'relu', padding = 'valid') )     #kan for some reason ikke definere dem udenfor og tilføje dem flere gange, dræber noget input
        
        # model.add(layers.Conv3D(convDepth[nr], size, padding='valid'))
        # model.add(layers.SpectralNormalization(layers.Dense(1)))  # Add spectral norm
        # model.add(layers.Activation('relu'))


        model.add(layers.MaxPooling3D(pool_size = (2,2,2)))
        if fullDR:
            model.add(layers.Dropout(rate = DropoutRate))
        #   model.add(layers.SpatialDropout3D(rate = DropoutRate))

    model.add(layers.GlobalAveragePooling3D()) if GAP else model.add(layers.Flatten())     #Enten flatterner eller global pooler
    if fullDR:
        model.add(layers.Dropout(rate = DropoutRate))

    for i in denseWidth:                                #Adder dense layers
        model.add(layers.Dense(i, activation='relu'))
        if fullDR:
            model.add(layers.Dropout(rate = DropoutRate))   #ved ikke om man skal gøre så den ikke adder dropout for første dense layer da det allerede er gjort længere nede




    if uncertaintyEst == 'nlll':                        #Hvis nlll bruges til uncertainty estimation skal vi have to output neuroner per variabel
        model.add(layers.Dense(vary_number * 2))        #Antal output parametre skal svare til antal varierende parametre ( * 2 her da nlll)
    elif uncertaintyEst == 'MC':
        model.add(layers.Dropout(rate = DropoutRate))
        model.add(layers.Dense(vary_number))
        # model.add(layers.SpectralNormalization(layers.Dense(vary_number)))

    return model

def make_bayesian_model(augmentation = False, convLayers = 1, convSize = None, convDepth = None, GAP = True, denseLayers = 1, BConv3D = True, BOutput = True):
    """Function to generate a Bayesian tensorflow model
    args:
    ------------------
    normalisering : bool
        Whether there should be a normalizing layer or not

    augmentation : bool
        Whether there should be a data augmentation layer (here rotation & random flips) or not

    convLayers : int
        Amount of convolutional layers

    convSize : array/list?
        Sizes of the convolutional layers, leave unspecificed or make sure its length is convLayers
    
    convDepth : array/list?
        Number of filters in each convolutional layer

    GAP : bool
        Whether to use GlobalAveragePooling or not (uses flatten if False)

    denseLayers : int
        Amount of dense layers (excluding final output layer)

    BConv3D : bool
        If Conv3D should be Bayesian or normal layer
    
    BOutput : bool
        If final output layer should be Bayesian or normal
    """
    if convSize == None:
        convSize = [(3,3,3)] * convLayers
    elif len(convSize) != convLayers:                           #Check at alle conv layers har en tilhørende size, ellers giv error
        raise ValueError('convSize must have length convLayers')
        
    if convDepth == None:                                       #Hvis ikke specified convDepth så bare lav standard for dem alle
        convDepth = [32] * convLayers
    elif len(convDepth) != convLayers:                           #Check at alle conv layers har en tilhørende depth, ellers giv error
        raise ValueError('convDepth must have length convLayers')

    model = models.Sequential([
    layers.Input(shape = (N,N,N)),                          #Giver input size, det vil den for some reason hellere have for sig ind i reshape
    layers.Reshape((N, N, N, 1))])
    
    # if normalisering:                                           #Med eller uden normalisering
    #     normalizer = layers.Normalization(axis = None)
    #     normalizer.adapt(X_train)
    #     model.add(normalizer)

    for nr, size in enumerate(convSize):                                       #Adder convolutional layers
        if BConv3D:                                                             #Bayesian eller normal conv3D, kan give bedre resultater at bruge normal
            model.add(tfpl.Convolution3DReparameterization(convDepth[nr], size, activation = 'relu', padding = 'valid') )     #kan for some reason ikke definere dem udenfor og tilføje dem flere gange, dræber noget input
        else:
            model.add(layers.Conv3D(convDepth[nr], size, activation = 'relu', padding = 'valid'))
        model.add(layers.MaxPooling3D(pool_size = (2,2,2)))


    model.add(layers.GlobalAveragePooling3D()) if GAP else model.add(layers.Flatten())     #Enten flatterner eller global pooler

    for _ in range(denseLayers):                                #Adder dense layers
        model.add(tfpl.DenseVariational(64, posterior, prior, kl_weight=1/len(y_train), kl_use_exact=True, activation='relu'))

    if BOutput:                                                                 #Bayesian eller normal output, kan give bedre resultater at bruge normal
        model.add(tfpl.DenseVariational(vary_number, posterior, prior, kl_weight=1/len(y_train), kl_use_exact=True))
    else:
        model.add(layers.Dense(vary_number))

    return model

def fit_model(model, trainGenerator, valGenerator, ReduceLROnPlateau = False, EarlyStopping = False, LearningRate = 1e-3, epochs = 50, uncertaintyEst = None):
    """Function to fit a Tensorflow model
    args:
    ----------------
    model : TF model
        Tensorflow model to fit

    ReduceLROnPlateau : bool
        Whether to reduce learning rate when loss flattens or not (helps prevent sudden spikes in loss in otherwise flat regimes)

    EarlyStopping : bool
        Whether to stop training when val loss flattens (note that ReduceLROnPlateau takes priority if both are enabled)
        
    LearningRate : float
        Initial learning rate for the optimizer
    
    epochs : int
        amount of epochs in training
    
    batch_size : int
        batch size for gradient descent, increase might stabilize loss function at the cost of performance
    """
    optimizer = optimizers.Adam(learning_rate = LearningRate)
    checkpoint = callbacks.ModelCheckpoint(model_savepath, monitor="val_loss", mode="min", save_best_only=True, verbose=0, save_weights_only = True)  #Kan gemme bedste weights med et callback ala det her i think, tænker ikke det er super relevant pt. koster måske også noget speed https://pub.aimind.so/never-use-restore-best-weights-true-with-earlystopping-754ba5f9b0c6
    
    if uncertaintyEst == None:                                                           #Skal have en måde at estimere usikkerhed på, ændrer compile en smule afhængig af hvad vi vil
        raise ValueError("uncertaintyEst must be either 'nlll', 'MC' or 'Bayesian'")
    elif uncertaintyEst == 'nlll':
        model.compile(optimizer = optimizer, loss = loss_nlll)
    else:
        model.compile(optimizer = optimizer, loss = loss_mse)    

    if ReduceLROnPlateau:
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr = 1e-5)              #reducerer learning rate hvis val_loss flader ud https://keras.io/api/callbacks/reduce_lr_on_plateau/, andre options vil være tf.clip_by_norm eller LearningRateScheduler
        history = model.fit(trainGenerator, epochs = epochs, validation_data = valGenerator, verbose = 0, callbacks = (reduce_lr, checkpoint))     #Callbacks er noget den gør under træning, så f.eks. efter hver epoch (ved ikke helt hvordan man indstiller hvor ofte?)
    
    elif EarlyStopping:
        earlstop = callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min')#, min_delta = 1e-3)                                 #Stopper learning hvis val_loss når et minimum, restore best weights er måske en trap https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
        history = model.fit(trainGenerator, epochs = epochs, validation_data = valGenerator, verbose = 0, callbacks = (earlstop, checkpoint))
    
    else:
        history = model.fit(trainGenerator, epochs = epochs, validation_data = valGenerator, verbose = 2, callbacks = checkpoint)
    
    model.load_weights(model_savepath)                                                  #Henter de bedste vægte
    export_dir = os.path.join(BASE_DIR, "full_model")
    model.save(export_dir)                     # gemmer både arkitektur og vægte
    print("Saved full model to", export_dir)

    return model.history.history

def test_og_plot_models(trainGenerator, valGenerator, testsamples = 10,
                        augmentation = False, convLayers = 1, convSize = None, convDepth = None, GAP = True, denseLayers = 1, DropoutRate = 0.2, fullDR = False, denseWidth = None,
                        ReduceLROnPlateau = False, EarlyStopping = False, LearningRate = 1e-3, epochs = 50, uncertaintyEst = None, BConv3D = True, BOutput = True,
                        ):
    """Function to create, fit and plot results for a model
    See make_model and fit_model for arg explanation
    """
    fig, ax = plt.subplots(2,2)
    if uncertaintyEst == 'Bayesian':                                                        #Hvis der vil bruges Bayesian uncertainty estimation så brug en Bayesian model, ellers så bare brug normal, evt. med dropout
        model = make_bayesian_model(augmentation = augmentation, convLayers = convLayers, convSize = convSize, convDepth = convDepth, GAP = GAP, denseLayers = denseLayers, BConv3D = BConv3D, BOutput = BOutput)
    else:
        model = make_model(augmentation = augmentation, convLayers = convLayers, convSize = convSize, convDepth = convDepth, GAP = GAP, denseLayers = denseLayers, DropoutRate = DropoutRate, fullDR = fullDR, uncertaintyEst = uncertaintyEst, denseWidth = denseWidth)           #Lader til at den arbejder videre med samme model hvis man ikke laver en ny hver gang
    

    startTrainingTime = time.time()
    history = fit_model(model = model, trainGenerator = trainGenerator, valGenerator = valGenerator, epochs = epochs, ReduceLROnPlateau = ReduceLROnPlateau, EarlyStopping = EarlyStopping, LearningRate = LearningRate, uncertaintyEst = uncertaintyEst)
    TrainTime = (time.time() - startTrainingTime)                                           #Gemmer tiden det har taget at træne netværket til sammenligning
    # 2) build & make the export dir
    export_dir = os.path.join(BASE_DIR, "full_model")
    os.makedirs(export_dir, exist_ok=True)
    print("→ Saving full model into:", export_dir)

    # 3) actually save; this will write saved_model.pb + variables/
    model.save(export_dir)
    print("✅ Full SavedModel written to", export_dir)

    loss = np.array(history['loss'])                                                        #Gemmer loss til senere plotning
    val_loss = np.array(history['val_loss'])

    x = np.arange(1, len(loss) + 1)

    index = (np.arange(len(y_test)) + 1)                                                    #Tager random samples fra vores testsæt, vil nok være pænere med en dictionary
    random.shuffle(index)
    X_test = [np.load(path + f'Test data//delta_test_id-{i}.npy') for i in index[: testsamples]]
    X_test = np.array(X_test)

    if uncertaintyEst == 'nlll':                                                            #Doesnt work for now, skal have ændret loss_nlll hvis jeg vil have den til at virke (tager kun en parameter rn)
        sigguess = model(X_test).numpy().T                                                  #nlll fikset men ikke tilpasset nyt dataformat so still shouldnt work xd
        A_mu = sigguess[0] * A_s_max        ; A_err = tf.math.softplus(sigguess[1]).numpy() * A_s_max
        n_mu = sigguess[2] * n_s_max        ; n_err = tf.math.softplus(sigguess[3]).numpy() * n_s_max
        o_mu = sigguess[4] * omega_cdm_max  ; o_err = tf.math.softplus(sigguess[5]).numpy() * omega_cdm_max

    # kig evt på at bruge predictions = [model.predict(X_test, batch_size=32, verbose=0) for _ in range(tests)] her
    else:                                                                                   #MC dropout lav dem evt. som seperate funktioner, vil nok være lidt pænere tbh
        tests = 50                                                                          #Antal gange modellen skal virke på samme R (større giver bedre estimat af usikkerhed men tager længere tid)
        A_mu = np.zeros(len(X_test)) ; A_err = np.zeros(len(X_test))                        #Initializer vores predictions
        n_mu = np.zeros(len(X_test)) ; n_err = np.zeros(len(X_test))
        o_mu = np.zeros(len(X_test)) ; o_err = np.zeros(len(X_test))

        scales = np.array([A_s_max, n_s_max, omega_cdm_max])
        for nrR, R in enumerate(X_test):                                                    #Fra deepseek, Gør det samme som egen implementering bare i batches så man slipper for dobbelt loop
            # Create batch of repeated R values (shape: [tests, ...])
            batch_R = np.repeat(R[np.newaxis, ...], tests, axis=0)
            
            # Get all predictions in a single forward pass (no .T needed)
            sigguesses = model(batch_R, training=True).numpy()  # shape: (tests, vary_number)
            
            # Initialize and populate parameters (vectorized)
            params = np.zeros((tests, 3))
            params[:, vary_flags] = sigguesses  # assumes vary_flags matches model output
            
            # Scale all parameters in one operation
            scaled_params = params * scales  # broadcasting
            
            # Compute statistics (vectorized)
            A_mu[nrR], A_err[nrR] = scaled_params[:, 0].mean(), scaled_params[:, 0].std(ddof=1)
            n_mu[nrR], n_err[nrR] = scaled_params[:, 1].mean(), scaled_params[:, 1].std(ddof=1)
            o_mu[nrR], o_err[nrR] = scaled_params[:, 2].mean(), scaled_params[:, 2].std(ddof=1)



    export_dir = os.path.join(BASE_DIR, "full_model")
    model.save(export_dir)
    print("Saved full model to", export_dir)


    A_test = np.array([y_test[: , 0][i - 1] for i in index[: testsamples]]) * A_s_max ; n_test = np.array([y_test[: , 1][i - 1] for i in index[: testsamples]]) * n_s_max ; o_test = np.array([y_test[: , 2][i - 1] for i in index[: testsamples]]) * omega_cdm_max

    ax[0][0].scatter(x, loss, c = 'k', label = 'Loss')                                                          ; ax[0][0].scatter(x, val_loss, c = 'r', label = 'Val loss')
    ax[0][1].errorbar(A_test[::10], A_mu[::10], yerr = A_err[::10], capsize = 3, marker = 'o', linestyle = 'None', c = 'cyan')    ; ax[0][1].plot(A_test, A_test, c = 'b')
    ax[1][0].errorbar(n_test, n_mu, yerr = n_err, capsize = 3, marker = 'o', linestyle = 'None', c = 'cyan')    ; ax[1][0].plot(n_test, n_test, c = 'b')
    ax[1][1].errorbar(o_test, o_mu, yerr = o_err, capsize = 3, marker = 'o', linestyle = 'None', c = 'cyan')    ; ax[1][1].plot(o_test, o_test, c = 'b')

    
    ax[0][0].set(yscale = 'symlog', xlabel = 'Epoch') ; ax[0][1].set(xlabel = 'Actual  $A_s$', ylabel = 'predicted $A_s$') ; ax[1][0].set(xlabel = 'Actual $n_s$', ylabel = 'Predicted $n_s$') ; ax[1][1].set(xlabel = r'Actual $\Omega_{cdm}$', ylabel = r'Predicted $\Omega_{cdm}$')#, title = 'Predicted vs actual sigma')
    ax[0][0].legend() #; ax[0][0].legend() ; ax[1][0].legend()

    fig.suptitle(f'Training time = {round(TrainTime, 2)} s,   LR*1e3:   {LearningRate * 1e3}')
    plt.savefig(figpath + f'N={N}, N_sam = {N_samples}, Tt={round(TrainTime)}, BS={batch_size}, DnsLay={denseLayers}, DnsW={denseWidth}, CLLay={convLayers}, CLDep={convDepth}, CLS={convSize}, fDR={fullDR}, DRR={DropoutRate}, UNC={uncertaintyEst}.png', bbox_inches = 'tight')


    coverage1 = sigma_coverage_calc(y_expected = A_test, y_measured = A_mu, y_err = A_err, sigmaMult = 1)
    coverage2 = sigma_coverage_calc(y_expected = A_test, y_measured = A_mu, y_err = A_err, sigmaMult = 2)
    coverage3 = sigma_coverage_calc(y_expected = A_test, y_measured = A_mu, y_err = A_err, sigmaMult = 3)
    coverage = np.array([coverage1, coverage2, coverage3])
    A_MSE = abs_percent_deviation_calc(y_expected = A_test, y_measured = A_mu)

    print('Done :)')
    return A_mu, A_err, n_mu, n_err, o_mu, o_err, coverage, A_MSE



path = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Træningsdata fra Simon//"
os.makedirs(path, exist_ok=True)   # sikre, at mappen findes

# Filnavn for checkpoints
BASE_DIR = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN//"
model_savepath = os.path.join(BASE_DIR, "checkpoint.weights.h5")
figpath = os.path.join(path, "figures")
os.makedirs(figpath, exist_ok=True)


# path = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Træningsdata fra Simon\\"
# figpath = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Træningsdata fra Simon\figures\\"
# model_savepath = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\Træningsdata fra Simon\\"

z, L, N = np.loadtxt(path + 'BoxParams.txt', dtype = 'int')                                                             #Loader dem for at sikre de passer med det data der er generated
A_s_min, A_s_max, n_s_min, n_s_max, omega_cdm_min, omega_cdm_max, N_samples = np.loadtxt(path + 'MaxMinParams.txt')     #Loader brugte max og min values for normalisering
vary_flags, vary_number = get_variable_params(A_s_min, A_s_max, n_s_min, n_s_max, omega_cdm_min, omega_cdm_max)         #Giver hvilke og hvor mange parametre der varierer
y_train, y_val, y_test = get_data(path)                                                                                 #Henter lavet data
N_samples = len(y_train) + len(y_val)                                                                                   #In case dataet ikke blev færdig med at generere, generelt mere stabilt tbh
partition, labels = make_partition_and_labels(N_samples = N_samples, y_train = y_train, y_val = y_val)

print(f'Varying params [A_s, n_s, omega_cdm] = {vary_flags}')
print(f'{z = }, {L = }, {N = }, {N_samples = }')                                                                                                    #Printer bare et par values for at kunne se dem når den kører
print(f'min/max \t A_s = ({A_s_min},{A_s_max}), n_s = ({n_s_min}, {n_s_max}), Omega_cdm  = ({omega_cdm_min}, {omega_cdm_max})')


batch_size = 10
params = {'dim': (N, N, N),
          'batch_size' : batch_size,
          'n_channels': 1,}
training_generator = DataGenerator(partition['train'], labels, **params, shuffle = True, augmentation = True)
validation_generator = DataGenerator(partition['validation'], labels, **params, shuffle = False, augmentation = False)

#68,95,99.7

ts = len(y_test)

"""Nogle bayesian tests"""
# _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1500, uncertaintyEst = 'Bayesian', LearningRate = 4e-3)
# print(f'Coverage: {coverage}')
# print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')

# _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1000, uncertaintyEst = 'Bayesian', LearningRate = 4e-3, BConv3D = False)
# print(f'Coverage: {coverage}')
# print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')

# _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 500, uncertaintyEst = 'Bayesian', LearningRate = 4e-3, BOutput = False)
# print(f'Coverage: {coverage}')
# print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')

# _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 500, uncertaintyEst = 'Bayesian', LearningRate = 4e-3, BConv3D = False, BOutput = False)
# print(f'Coverage: {coverage}')
# print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')

"""nogle MC tests"""
# _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 500, uncertaintyEst = 'MC', LearningRate = 4e-3, fullDR = False)
# _, _, _, _, _, _, coverage_fDR, p_dev_fDR = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 500, uncertaintyEst = 'MC', LearningRate = 4e-3, fullDR = True)
# print(f'Coverage: {coverage}')
# print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')
# print(f'Coverage_fDR: {coverage_fDR}')
# print(f'Percent deviation for predictions: \n ({round(p_dev_fDR[0], 2)} +- {round(p_dev_fDR[1], 2)}) %')

def main():
    # alt det du skrev på bunden: load data, parser args, 
    # opbyg generators, test_og_plot_models(...) osv.
    _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(
        training_generator,
        validation_generator,
        testsamples=ts,
        epochs=args.epochs,
        uncertaintyEst='MC',
        LearningRate=4e-3,
        fullDR=False
    )
    plt.show()

if __name__ == "__main__":
    main()

"""


Tror apparently også at man kan lave to seperate netværk i en for flere output predictions? - ok ig det bare er at lave en class med to NN i sig https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

Loss med multiple outputs: https://keras.io/api/losses/
returnerer vist average loss for de forskellige parametre basically (sum_over_batch_size er default)

INDFØR SHUFFLE MELLEM EPOCHS - KAN HJÆLPE SGD?

depthwise conv
https://www.reddit.com/r/MLQuestions/comments/gp2pj9/what_is_depthwiseconv2d_and_separableconv2d_in/

memory
https://stackoverflow.com/questions/46066850/understanding-the-resourceexhaustederror-oom-when-allocating-tensor-with-shape
https://medium.com/analytics-vidhya/reducing-deep-learning-model-size-without-effecting-its-original-performance-and-accuracy-with-a809b49cf519
https://www.reddit.com/r/tensorflow/comments/13vrs25/how_can_i_reduce_the_size_of_my_machine_learning/
https://www.tensorflow.org/tutorials/optimization/compression
https://www.omi.me/blogs/tensorflow-errors/out-of-memory-in-tensorflow-causes-and-how-to-fix

https://forums.fast.ai/t/training-4-5-million-parameters-takes-30-mins-per-epoch/83081/11
"""



# #%% Opsætter test af multithreading om det overhovedet gør noget bare smidt det her ind i en .pyfil ved siden af COsmoN4.py og kør det
# import os
# import time
# import psutil
# import subprocess
# import statistics
# import sys

# VENV_PY = r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN\venvNN\Scripts\python.exe"


# def benchmark(script_path, core_list, omp_threads, affinity, epochs=50):
#     """
#     Kør CosmoN4.py med:
#       - kun kerner i core_list tilgængelige
#       - omp_threads OpenMP-tråde
#       - KMP_AFFINITY som angivet
#     Returnerer total køretid i sekunder.
#     """
#     env = os.environ.copy()
#     env["OMP_NUM_THREADS"] = str(omp_threads)
#     env["KMP_AFFINITY"]    = affinity
#     env["KMP_SETTINGS"]    = "1"

#     cmd = [
#         VENV_PY,
#         script_path,
#         f"--epochs={epochs}"
#     ]
    
#     # 2) Start CosmoN4.py som en ny proces
#     proc = subprocess.Popen(
#         cmd,
#         env=env,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         text=True,
#         bufsize = 1,
        
#     )

#     # 3) Bind affinity på selve CosmoN4-processen
#     p = psutil.Process(proc.pid).cpu_affinity(core_list)



#     t0 = time.perf_counter()
#     for line in proc.stdout:
#         print(line, end="")  

#     proc.wait()
#     dt = time.perf_counter() - t0
#     return dt

# def repeat_benchmark(script_path, core_list, omp_threads, affinity,
#                     repeats=2, epochs=50):
#     times = []
#     for i in range(1, repeats+1):
#         print(f"\n  → Run {i}/{repeats} for cores={core_list}, threads={omp_threads}, epochs={epochs}")
#         dt = benchmark(script_path, core_list, omp_threads, affinity, epochs=epochs)
#         print(f"    [Done in {dt:.3f}s]")
#         times.append(dt)
#     return statistics.mean(times), statistics.stdev(times) if repeats>1 else 0.0

# if __name__ == "__main__":
#     script = r"C:/Users/habbo/OneDrive - Aarhus universitet/Skole/UNI/6. Sem/BachelorProjekt/Data/Different_input_NN/COsmo N4.py"

#     # Liste over (kerne-subset, omp_threads, affinity) du vil teste:
#     tests = [
#         ([0],          1,  "granularity=fine,scatter,1,0",50), # 1 tråd på 1 kerne
#         (list(range(16)),  1,  "granularity=fine,scatter,1,0",50),# 16 kerner × 1 tråd
#         ([0],         16,  "granularity=fine,compact,16,0",50),# 1 kerne × 16 tråde
#         ([0,1,2,3],    4,  "granularity=fine,scatter,1,0",50),# 1 tråd på hver kerne (f.eks. kerner 0–3)
#         ([0,1],        4,  "granularity=fine,compact,2,0",50),# 2 tråde pr. kerne på 2 kerner
#         ([0,1,2,3],   16,  "granularity=fine,compact,4,0",50),# 4 tråde pr. kerne på 4 kerner
#     ]

#     results = []
#     print("\n=== Starter benchmark-sekvens ===\n")
#     for cores, threads, affinity, epochs in tests:
#         print(f"\n=== Tester Cores={cores}, Threads={threads}, Epochs={epochs} ===")
#         mean, std = repeat_benchmark(script, cores, threads, affinity, epochs=epochs)
#         results.append((cores, threads, epochs, mean, std))    
        
    
#     # Print oversigt
#     print("\nResultater:")
#     print("\n=== Resultater (gennemsnit ± std) ===")
#     print(f"{'Cores':20s}  {'Thr':>3s}   {'Mean (s)':>8s}   {'Std (s)':>7s}")
#     print("-"*45)
#     for cores, thr, aff, m, s in results:
#         cores_str = ",".join(map(str, cores))
#         print(f"{cores_str:20s}   {thr:3d}   {m:8.3f}   ± {s:6.3f}")
# %%
# #%% Pipeline til at få data fra delta-felter

# base = Path(r"C:\Users\habbo\OneDrive - Aarhus universitet\Skole\UNI\6. Sem\BachelorProjekt\Data\Different_input_NN")

# # 2) Liste over alle tekstfiler med Delta_AS=…    
# txt_paths = sorted(base.glob("NN_As*/*/snapshot/delta_Mtxt/Delta_AS=*.txt"))

# # 3) Ekstraher labels (A_s-værdier) fra filnavne
# labels = np.array([float(p.stem.split("=")[1]) for p in txt_paths], dtype=np.float32)

# # 4) Loader funktionen til tf.data
# def _load_delta(path):
#     arr = np.loadtxt(path.decode(), dtype=np.float32)
#     N = int(round(arr.size ** (1/3)))
#     vol = arr.reshape((N, N, N, 1))
#     return vol

# def parse_fn(path, label):
#     vol = tf.numpy_function(_load_delta, [path], tf.float32)
#     vol.set_shape([None, None, None, 1])
#     return vol, label

# # 5) Byg dataset    
# dataset = (
#     tf.data.Dataset
#       .from_tensor_slices(( [str(p) for p in txt_paths], labels ))
#       .shuffle(len(txt_paths))
#       .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
#       .batch(16)
#       .prefetch(tf.data.AUTOTUNE)
# )

# # 6) (Evt. test ét batch)
# for x_batch, y_batch in dataset.take(1):
#     print("Batch shapes:", x_batch.shape, y_batch.shape)

#%%