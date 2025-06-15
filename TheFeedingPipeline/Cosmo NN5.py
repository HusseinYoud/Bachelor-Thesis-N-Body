"""Imports & setting up træningdata"""
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", message=".*layer.add_variable.*")          #Tror ikke umiddelbart de her warnings er noget der kan gøres noget ved
warnings.filterwarnings("ignore", message=".*RandomNormal is unseeded.*")         #Tror ikke umiddelbart de her warnings er noget der kan gøres noget ved, måske der kan ved den her?
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                   #Fjerner en notice om noget numerisk precision or smth, gør det til gengæld også like 100% langsommere & might just kill everything so ye, dont do this
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                    #Fjerner notice om at CPU bliver brugt til at optimize stuff, kan måske fjerne relevante ting også not sure so be careful
import tensorflow as tf
from tf_keras import layers, models, optimizers, callbacks, utils, backend
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import psutil
import random
plt.rc("axes", labelsize=30, titlesize=32)   # skriftstørrelse af xlabel, ylabel og title
plt.rc("xtick", labelsize=26, top=True, direction="in")  # skriftstørrelse af ticks, vis også ticks øverst og vend ticks indad
plt.rc("ytick", labelsize=26, right=True, direction="in") # samme som ovenstående
plt.rc("legend", fontsize=30) # skriftstørrelse af figurers legends
plt.rcParams["font.size"] = "20"
plt.rcParams["figure.figsize"] = (16,9)

def loss_mse(y_true, y_pred):
    """Standard mean squared error loss function for Tensorflow to use & abuse"""
    y_pred = tf.squeeze(y_pred)  # Removes singleton dimensions → shape (batch_size,)
    y_true = tf.squeeze(y_true)  # Den anbefaler at bruge dem for robustness, kan også specify [: , 0] på dem begge eller bare undlade det

    # print(f'{y_pred = }')     # for deubbigng
    # print(f'{y_true = }')

    loss = (y_pred - y_true)**2         #Tror bare den tager det her for hver individuel komponent
    return tf.reduce_mean(loss)  #tf.reduce_mean minder om (måske nøjagtig samme?) som np.mean()  - https://stackoverflow.com/questions/34236252/what-is-the-difference-between-np-mean-and-tf-reduce-mean

def loss_nlll(y_true, y_pred, eps = 1e-6):
    "Generalizd Gaussian negative log likelihood  to accomodate more than one parameter"
    losses = ([])
    for j in range(vary_number):
        i = 2 * j
        mu = y_pred[:, i]
        sig_raw = y_pred[:, i+1]

        sigma = sigma = tf.math.softplus(sig_raw) + eps
        var = sigma**2

        mu = tf.squeeze(mu) ; var = tf.squeeze(var)
        y_mu = tf.squeeze(y_true[: , j])

        temp_loss = 0.5 * (tf.math.log(var) + (y_mu - mu)**2 / var)
        losses.append(temp_loss)


    tot_loss = tf.add_n(losses)                                
    return tf.reduce_mean(tot_loss)


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
    abs_deviation = abs(y_expected - y_measured) #/ y_expected
    dev_mu = np.mean(abs_deviation) #* 100
    dev_std = np.std(abs_deviation, ddof = 1) #* 100

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
    """Finds which parameters vary and how many there are"""
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
            if Xi.shape == self.dim:
                Xi = Xi[..., np.newaxis]                                                #Sometimes input data is (N,N,N) instead of (N,N,N,1)

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
        model.add(layers.Dropout(rate = DropoutRate))
        model.add(layers.Dense(vary_number * 2))        #Antal output parametre skal svare til antal varierende parametre ( * 2 her da nlll)

    elif uncertaintyEst == 'MC':
        model.add(layers.Dropout(rate = DropoutRate))
        model.add(layers.Dense(vary_number))
        # model.add(layers.SpectralNormalization(layers.Dense(vary_number)))

    return model

def make_bayesian_model(augmentation = False, convLayers = 1, convSize = None, convDepth = None, GAP = True, denseLayers = 1, BConv3D = True, BOutput = True, BDense = True):
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
        if BDense:
            model.add(tfpl.DenseVariational(64, posterior, prior, kl_weight=1/len(y_train), kl_use_exact=True, activation='relu'))
        else:
            model.add(layers.Dense(64, activation='relu'))

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
        history = model.fit(trainGenerator, epochs = epochs, validation_data = valGenerator, verbose = 2, callbacks = (reduce_lr, checkpoint))     #Callbacks er noget den gør under træning, så f.eks. efter hver epoch (ved ikke helt hvordan man indstiller hvor ofte?)
    
    elif EarlyStopping:
        earlstop = callbacks.EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'min', start_from_epoch = 300)#, min_delta = 1e-3)                                 #Stopper learning hvis val_loss når et minimum, restore best weights er måske en trap https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
        history = model.fit(trainGenerator, epochs = epochs, validation_data = valGenerator, verbose = 2, callbacks = (earlstop, checkpoint))
    
    else:
        history = model.fit(trainGenerator, epochs = epochs, validation_data = valGenerator, verbose = 2, callbacks = checkpoint)
    
    model.load_weights(model_savepath)                                                  #Henter de bedste vægte

    return model.history.history

def test_og_plot_models(trainGenerator, valGenerator, testsamples = 10,
                        augmentation = False, convLayers = 1, convSize = None, convDepth = None, GAP = True, denseLayers = 1, DropoutRate = 0.2, fullDR = False, denseWidth = None,
                        ReduceLROnPlateau = False, EarlyStopping = False, LearningRate = 1e-3, epochs = 50, uncertaintyEst = None, BConv3D = True, BOutput = True, BDense = True
                        ):
    """Function to create, fit and plot results for a model
    See make_model and fit_model for arg explanation
    """
    fig, ax = plt.subplots(2,2)
    if uncertaintyEst == 'Bayesian':                                                        #Hvis der vil bruges Bayesian uncertainty estimation så brug en Bayesian model, ellers så bare brug normal, evt. med dropout
        model = make_bayesian_model(augmentation = augmentation, convLayers = convLayers, convSize = convSize, convDepth = convDepth, GAP = GAP, denseLayers = denseLayers, BConv3D = BConv3D, BOutput = BOutput, BDense = BDense)
    else:
        model = make_model(augmentation = augmentation, convLayers = convLayers, convSize = convSize, convDepth = convDepth, GAP = GAP, denseLayers = denseLayers, DropoutRate = DropoutRate, fullDR = fullDR, uncertaintyEst = uncertaintyEst, denseWidth = denseWidth)           #Lader til at den arbejder videre med samme model hvis man ikke laver en ny hver gang
    

    startTrainingTime = time.time()
    history = fit_model(model = model, trainGenerator = trainGenerator, valGenerator = valGenerator, epochs = epochs, ReduceLROnPlateau = ReduceLROnPlateau, EarlyStopping = EarlyStopping, LearningRate = LearningRate, uncertaintyEst = uncertaintyEst)
    TrainTime = (time.time() - startTrainingTime)                                           #Gemmer tiden det har taget at træne netværket til sammenligning

    loss = np.array(history['loss'])                                                        #Gemmer loss til senere plotning
    val_loss = np.array(history['val_loss'])

    x = np.arange(1, len(loss) + 1)

    index = (np.arange(len(y_test)) + 1)                                                    #Tager random samples fra vores testsæt, vil nok være pænere med en dictionary
    random.shuffle(index)
    X_test = [np.load(path + f'Test data//delta_test_id-{i}.npy') for i in index[: testsamples]]
    X_test = np.array(X_test)

    tests = 50                                                                          #Antal gange modellen skal virke på samme R (større giver bedre estimat af usikkerhed men tager længere tid)
    A_mu = np.zeros(len(X_test)) ; A_err = np.zeros(len(X_test)) ; A_err_al = np.zeros(len(X_test))                        #Initializer vores predictions
    n_mu = np.zeros(len(X_test)) ; n_err = np.zeros(len(X_test)) ; n_err_al = np.zeros(len(X_test))
    o_mu = np.zeros(len(X_test)) ; o_err = np.zeros(len(X_test)) ; o_err_al = np.zeros(len(X_test))

    for nrR, R in enumerate(X_test):
        batch_R = np.repeat(R[np.newaxis, ...], tests, axis=0)                                      #vectorized with help by deepseek to prevent double looping

        sigguesses = model(batch_R, training=True).numpy()                  #Batch predicting like this does not seem to work for estimating uncertainty when using bayesian models, i presume the seed used for sampling doesnt change between different R's which causes this. So use slow double loop for Baysian model if relevant ig.
        if uncertaintyEst == 'nlll':
            params = np.zeros((tests, 3 * 2))
            params[:, np.repeat(vary_flags, 2)] = sigguesses
            A_mu_temp = params[:, 0] * A_s_max        ; A_err_temp = tf.math.softplus(params[:, 1]).numpy() * A_s_max
            n_mu_temp = params[:, 2] * n_s_max        ; n_err_temp = tf.math.softplus(params[:, 3]).numpy() * n_s_max
            o_mu_temp = params[:, 4] * omega_cdm_max  ; o_err_temp = tf.math.softplus(params[:, 5]).numpy() * omega_cdm_max

            A_mu[nrR], A_err[nrR], A_err_al[nrR] = A_mu_temp.mean(), A_mu_temp.std(ddof=1), A_err_temp.mean()
            n_mu[nrR], n_err[nrR], n_err_al[nrR] = n_mu_temp.mean(), n_mu_temp.std(ddof=1), n_err_temp.mean()
            o_mu[nrR], o_err[nrR], o_err_al[nrR] = o_mu_temp.mean(), o_mu_temp.std(ddof=1), o_err_temp.mean()

        else:
            scales = np.array([A_s_max, n_s_max, omega_cdm_max])
            params = np.zeros((tests, 3))
            params[:, vary_flags] = sigguesses
            
            # Scale all parameters in one operation
            scaled_params = params * scales  # broadcasting
            
            # Compute statistics (vectorized)
            A_mu[nrR], A_err[nrR] = scaled_params[:, 0].mean(), scaled_params[:, 0].std(ddof=1)
            n_mu[nrR], n_err[nrR] = scaled_params[:, 1].mean(), scaled_params[:, 1].std(ddof=1)
            o_mu[nrR], o_err[nrR] = scaled_params[:, 2].mean(), scaled_params[:, 2].std(ddof=1)




    A_test = np.array([y_test[: , 0][i - 1] for i in index[: testsamples]]) * A_s_max ; n_test = np.array([y_test[: , 1][i - 1] for i in index[: testsamples]]) * n_s_max ; o_test = np.array([y_test[: , 2][i - 1] for i in index[: testsamples]]) * omega_cdm_max

    ax[0][1].errorbar(A_test[::1], A_mu[::1], yerr = A_err[::1], capsize = 3, marker = 'o', linestyle = 'None', c = 'cyan')    ; ax[0][1].plot(A_test, A_test, c = 'b')
    ax[1][0].errorbar(n_test, n_mu, yerr = n_err, capsize = 3, marker = 'o', linestyle = 'None', c = 'cyan')    ; ax[1][0].plot(n_test, n_test, c = 'b')
    ax[1][1].errorbar(o_test, o_mu, yerr = o_err, capsize = 3, marker = 'o', linestyle = 'None', c = 'cyan')    ; ax[1][1].plot(o_test, o_test, c = 'b')
    ax[0][0].scatter(x, loss, c = 'k', label = 'Loss')                                                          ; ax[0][0].scatter(x, val_loss, c = 'r', label = 'Val loss')
    if uncertaintyEst == 'nlll':
        ax[0][1].errorbar(A_test[::8], A_mu[::8], yerr = A_err_al[::8], capsize = 3, marker = 'o', linestyle = 'None', c = 'g', label = 'Aleatoric uncertainty')
        ax[1][0].errorbar(n_test, n_mu, yerr = n_err_al, capsize = 3, marker = 'o', linestyle = 'None', c = 'g', label = 'Aleatoric uncertainty')
        ax[1][1].errorbar(o_test, o_mu, yerr = o_err_al, capsize = 3, marker = 'o', linestyle = 'None', c = 'g', label = 'Aleatoric uncertainty')

    
    ax[0][0].set(yscale = 'symlog', xlabel = 'Epoch') ; ax[0][1].set(xlabel = 'Actual  $A_s$', ylabel = 'predicted $A_s$') ; ax[1][0].set(xlabel = 'Actual $n_s$', ylabel = 'Predicted $n_s$') ; ax[1][1].set(xlabel = r'Actual $\Omega_{cdm}$', ylabel = r'Predicted $\Omega_{cdm}$')#, title = 'Predicted vs actual sigma')
    ax[0][0].legend() ; ax[0][1].legend() ; ax[1][0].legend()

    fig.suptitle(f'Training time = {round(TrainTime, 2)} s,   LR*1e3:   {LearningRate * 1e3}')

    try:
        plt.savefig(figpath + f'N={N}, N_sam = {N_samples}, Tt={round(TrainTime)}, BS={batch_size}, DnsLay={denseLayers}, DnsW={denseWidth}, CLLay={convLayers}, CLDep={convDepth}, CLS={convSize}, fDR={fullDR}, DRR={DropoutRate}, UNC={uncertaintyEst}.png', bbox_inches = 'tight')
    except FileNotFoundError:
        print('Figure not saved, make sure to specify a path to an existing directory if you want to save the figures')
    else:
        print('Figure not saved, filepath found but something else went wrong')

    coverage1 = sigma_coverage_calc(y_expected = A_test, y_measured = A_mu, y_err = A_err, sigmaMult = 1)
    coverage2 = sigma_coverage_calc(y_expected = A_test, y_measured = A_mu, y_err = A_err, sigmaMult = 2)
    coverage3 = sigma_coverage_calc(y_expected = A_test, y_measured = A_mu, y_err = A_err, sigmaMult = 3)
    coverage = np.array([coverage1, coverage2, coverage3])
    A_MSE = abs_percent_deviation_calc(y_expected = A_test, y_measured = A_mu) * 10**9

    coverage1 = sigma_coverage_calc(y_expected = n_test, y_measured = n_mu, y_err = n_err, sigmaMult = 1)
    coverage2 = sigma_coverage_calc(y_expected = n_test, y_measured = n_mu, y_err = n_err, sigmaMult = 2)
    coverage3 = sigma_coverage_calc(y_expected = n_test, y_measured = n_mu, y_err = n_err, sigmaMult = 3)
    coverage_n = np.array([coverage1, coverage2, coverage3])
    n_MSE = abs_percent_deviation_calc(y_expected = n_test, y_measured = n_mu)


    del model, history      
    print('Done :)')

    print(f'{coverage = }')
    print(f'{A_MSE = }')

    # LPTpath = r'/home/candifloos/Bachelor/NN models/LPTdata//'  
    # with open(LPTpath + 'Predictions, z=0, Johnny, 3000 epochs.txt', mode = 'w') as f:
    #     f.write(f'True A_s \t Predicted A_s \t A_s error \n')
    #     for i in range(len(A_mu)):
    #         f.write(f'{A_test[i]} \t {A_mu[i]} \t {A_err[i]} \n')
    # np.save(LPTpath + 'loss, z=0, Johnny, 3000 epochs', loss)
    # np.save(LPTpath + 'val loss, z=0, Johnny, 3000 epochs', val_loss)

    return A_mu, A_err, n_mu, n_err, o_mu, o_err, coverage, A_MSE, coverage_n, n_MSE

def mk_bs_lr_tests(valtimes = 1):
    batch_sizes = np.array([1, 2, 4, 8, 12, 16])
    # LRs = np.array([0.5, 1, 2, 4, 8, 12, 20, 32, 64, 100, 150])    #*1e3
    LRs = np.geomspace(0.5, 200, 10)
    print(LRs)

    A_dev = np.zeros((len(batch_sizes), len(LRs)))
    times = np.zeros((len(batch_sizes), len(LRs)))
    coverages1 = np.zeros((len(batch_sizes), len(LRs)))
    epochs = 1500                 #Tilpas den her
    for nrb, bs in enumerate(batch_sizes):
        params['batch_size'] = bs
        training_generator = DataGenerator(partition['train'], labels, **params, shuffle = True, augmentation = True)
        validation_generator = DataGenerator(partition['validation'], labels, **params, shuffle = False, augmentation = False)

        for nrl, lr in enumerate(LRs):
            
            A_deviation_temp = np.zeros(valtimes)
            temp_times = np.zeros(valtimes)
            temp_cov = np.zeros(valtimes)

            for k in range(valtimes):
                startime = time.time()
                _, _, _, _, _, _, coverage, p_dev, coverage_n, p_dev_n = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = epochs, uncertaintyEst = 'MC', LearningRate = lr * 1e-3, fullDR = False, DropoutRate = 0.2, EarlyStopping = False)
                traintime = time.time() - startime
                A_deviation_temp[k] = p_dev[0]
                temp_times[k] = traintime
                temp_cov[k] = coverage[0]
                plt.close()

            times[nrb][nrl] = np.mean(temp_times)
            A_dev[nrb][nrl] = np.mean(A_deviation_temp)
            coverages1[nrb][nrl] = np.mean(temp_cov)

    if not os.path.exists(data_savepath + 'BS&LR'):
        os.mkdir(data_savepath + 'BS&LR')

    np.save(data_savepath + 'BS&LR/A_dev', A_dev)
    np.save(data_savepath + 'BS&LR/LRs', LRs)
    np.save(data_savepath + 'BS&LR/batch_sizes', batch_sizes)
    np.save(data_savepath + 'BS&LR/times', times)
    np.save(data_savepath + 'BS&LR/coverages', coverages1)

def mk_bs_epoch_tests(valtimes = 1):
    batch_sizes = np.array([1, 2, 4, 8, 12, 16])
    epochs = np.array([10, 25, 50, 100, 250, 500, 1000, 1500, 2000])

    A_dev = np.zeros((len(batch_sizes), len(epochs)))
    times = np.zeros((len(batch_sizes), len(epochs)))
    coverages1 = np.zeros((len(batch_sizes), len(epochs)))
    for nrb, bs in enumerate(batch_sizes):
        params['batch_size'] = bs
        batch_size = bs
        print(bs)
        training_generator = DataGenerator(partition['train'], labels, **params, shuffle = True, augmentation = True)
        validation_generator = DataGenerator(partition['validation'], labels, **params, shuffle = False, augmentation = False)

        for nrl, epoch in enumerate(epochs):
            A_deviation_temp = np.zeros(valtimes)
            temp_times = np.zeros(valtimes)
            temp_cov = np.zeros(valtimes)

            for k in range(valtimes):
                startime = time.time()
                _, _, _, _, _, _, coverage, p_dev, coverage_n, p_dev_n = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = epoch, uncertaintyEst = 'MC', LearningRate = 4 * 1e-3, fullDR = False, DropoutRate = 0.2, EarlyStopping = False)
                traintime = time.time() - startime
                A_deviation_temp[k] = p_dev[0]
                temp_times[k] = traintime
                temp_cov[k] = coverage[0]
                plt.close()

            times[nrb][nrl] = np.mean(temp_times)
            A_dev[nrb][nrl] = np.mean(A_deviation_temp)
            coverages1[nrb][nrl] = np.mean(temp_cov)

    if not os.path.exists(data_savepath + 'BS&Epochs'):
        os.mkdir(data_savepath + 'BS&Epochs')

    np.save(data_savepath + 'BS&Epochs/A_dev', A_dev)
    np.save(data_savepath + 'BS&Epochs/epochs', epochs)
    np.save(data_savepath + 'BS&Epochs/batch_sizes', batch_sizes)
    np.save(data_savepath + 'BS&Epochs/times', times)
    np.save(data_savepath + 'BS&Epochs/coverages', coverages1)

def mk_lr_epoch_tests(valtimes = 1):
    if params['batch_size'] != 16:
        print(f'Warning batch_size = {params["batch_size"]}, is this intented?')        #Give warning in case mk_bs_epoch has been called earlier and changed params unintentionally

    LRs = np.geomspace(0.1, 128, 10)
    epochs = np.geomspace(10, 2000, 10)

    A_dev = np.zeros((len(LRs), len(epochs)))
    times = np.zeros((len(LRs), len(epochs)))
    coverages1 = np.zeros((len(LRs), len(epochs)))
    for nrb, lr in enumerate(LRs):
        print(lr)
        for nrl, epoch in enumerate(epochs):
            epoch = int(epoch)
            A_deviation_temp = np.zeros(valtimes)
            temp_times = np.zeros(valtimes)
            temp_cov = np.zeros(valtimes)

            for k in range(valtimes):
                startime = time.time()
                _, _, _, _, _, _, coverage, p_dev, coverage_n, p_dev_n = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = epoch, uncertaintyEst = 'MC', LearningRate = lr * 1e-3, fullDR = False, DropoutRate = 0.2, EarlyStopping = False)
                traintime = time.time() - startime
                A_deviation_temp[k] = p_dev[0]
                temp_times[k] = traintime
                temp_cov[k] = coverage[0]
                plt.close()

            times[nrb][nrl] = np.mean(temp_times)
            A_dev[nrb][nrl] = np.mean(A_deviation_temp)
            coverages1[nrb][nrl] = np.mean(temp_cov)

    if not os.path.exists(data_savepath + 'lr&Epochs'):
        os.mkdir(data_savepath + 'lr&Epochs')

    np.save(data_savepath + 'lr&Epochs/A_dev', A_dev)
    np.save(data_savepath + 'lr&Epochs/epochs', epochs)
    np.save(data_savepath + 'lr&Epochs/LRs', LRs)
    np.save(data_savepath + 'lr&Epochs/times', times)
    np.save(data_savepath + 'lr&Epochs/coverages', coverages1)


path = r'/home/candifloos/Bachelor/NN models/Created data//'                            #path to the directory containing the training, validation and test data                            should correspond with Cosmo data creator savepath
model_savepath = r'/home/candifloos/Bachelor/NN models/checkpoint.weights.h5'           #where to save model weights                                                                        whereever, make sure its .weights.h5 though
figpath = r'/home/candifloos/Figuredump//'                                              #dumping place for all figures containing loss and predictions                                      (OPTIONAL)
data_savepath = r'/home/candifloos/Bachelor/NN models/Plotting/Plot data//'             #Where to save data for external plots, currently only relevant when making hyperparameter plots    (OPTIONAL)

z, L, N = np.loadtxt(path + 'BoxParams.txt', dtype = 'int')                                                             #Loader dem for at sikre de passer med det data der er generated
A_s_min, A_s_max, n_s_min, n_s_max, omega_cdm_min, omega_cdm_max, N_samples = np.loadtxt(path + 'MaxMinParams.txt')     #Loader brugte max og min values for normalisering
vary_flags, vary_number = get_variable_params(A_s_min, A_s_max, n_s_min, n_s_max, omega_cdm_min, omega_cdm_max)         #Giver hvilke og hvor mange parametre der varierer
y_train, y_val, y_test = get_data(path)                                                                                 #Henter lavet data
N_samples = len(y_train) + len(y_val)                                                                                   #In case dataet ikke blev færdig med at generere, generelt mere stabilt tbh
partition, labels = make_partition_and_labels(N_samples = N_samples, y_train = y_train, y_val = y_val)

print(f'Varying params [A_s, n_s, omega_cdm] = {vary_flags}')
print(f'{z = }, {L = }, {N = }, {N_samples = }')                                                                                                    #Printer bare et par values for at kunne se dem når den kører
print(f'min/max \t A_s = ({A_s_min},{A_s_max}), n_s = ({n_s_min}, {n_s_max}), Omega_cdm  = ({omega_cdm_min}, {omega_cdm_max})')

batch_size = 16
params = {'dim': (N, N, N),
          'batch_size' : batch_size,
          'n_channels': 1,}
training_generator = DataGenerator(partition['train'], labels, **params, shuffle = True, augmentation = True)
validation_generator = DataGenerator(partition['validation'], labels, **params, shuffle = False, augmentation = False)
ts = len(y_test)

"""Nogle bayesian tests"""
# for _ in range(1):
#     _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1500, uncertaintyEst = 'Bayesian', LearningRate = 4e-3)
#     print(f'Coverage: {coverage}')
#     print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')

#     _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1000, uncertaintyEst = 'Bayesian', LearningRate = 4e-3, BConv3D = False)
#     print(f'Coverage: {coverage}')
#     print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')

#     _, _, _, _, _, _, coverage, p_dev = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 500, uncertaintyEst = 'Bayesian', LearningRate = 4e-3, BOutput = False)
#     print(f'Coverage: {coverage}')
#     print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')

# _, _, _, _, _, _, coverage, p_dev, _, _ = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1000, uncertaintyEst = 'Bayesian', LearningRate = 4e-3, BConv3D = False, BOutput = False)
    # print(f'Coverage: {coverage}')
    # print(f'Percent deviation for predictions: \n ({round(p_dev[0], 2)} +- {round(p_dev[1], 2)}) %')

"""nogle MC tests"""
# for _ in range(1):
#     # convdepth = [64,]
#     _, _, _, _, _, _, coverage, p_dev, coverage_n, p_dev_n = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1500, uncertaintyEst = 'MC', LearningRate = 1e-3, fullDR = False, DropoutRate = 0.2, EarlyStopping = True)
#     # _, _, _, _, _, _, coverage_fDR, p_dev_fDR, coverage_n_fDR, p_dev_n_fDR = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 250, uncertaintyEst = 'MC', LearningRate = 4e-3, fullDR = True, GAP = True)
#     _, _, _, _, _, _, coverage_fDR, p_dev_fDR, coverage_n_fDR, p_dev_n_fDR = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 750, uncertaintyEst = 'MC', LearningRate = 4e-3, fullDR = True, EarlyStopping = True)

#     print(f'Coverage: {coverage}, {coverage_n}')
#     print(f'Deviation for predictions: \n A: ({round(p_dev[0], 3)} +- {round(p_dev[1], 3)}) n: ({round(p_dev_n[0], 3)} +- {round(p_dev_n[1], 3)})')

#     print(f'Coverage_fDR: {coverage_fDR}, {coverage_n_fDR}')
#     print(f'Deviation for predictions: \n ({round(p_dev_fDR[0], 3)} +- {round(p_dev_fDR[1], 3)}) n: ({round(p_dev_n_fDR[0], 3)} +- {round(p_dev_n_fDR[1], 3)})')
# A = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 10000, uncertaintyEst = 'Bayesian', LearningRate = 1e-3, BConv3D=False, BDense=False, BOutput=True)

# A = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1000, uncertaintyEst = 'MC', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.2)

# A = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1000, uncertaintyEst = 'nlll', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.2)

# A = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 25, uncertaintyEst = 'Bayesian', LearningRate = 4e-3, BConv3D=False, BDense=False, BOutput=True)
# A = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 25, uncertaintyEst = 'Bayesian', LearningRate = 4e-3, BConv3D=False, BDense=False, BOutput=False)
# B = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 3000, uncertaintyEst = 'MC', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.2)
# B = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 2000, uncertaintyEst = 'MC', LearningRate = 1e-3, fullDR = False, DropoutRate = 0.2)

# densewidth = [128, ]
# # B = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 250, uncertaintyEst = 'nlll', LearningRate = 1e-3, fullDR = False, DropoutRate = 0.2, denseWidth = densewidth, convLayers = 2)
# # A = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 250, uncertaintyEst = 'MC', LearningRate = 1e-3, fullDR = False, DropoutRate = 0.2, denseWidth = densewidth, convLayers = 2)
# # C = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 750, uncertaintyEst = 'nlll', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.2, denseWidth = densewidth)
# # C = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1500, uncertaintyEst = 'nlll', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.1, denseWidth = densewidth)
# # C = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1500, uncertaintyEst = 'nlll', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.2, denseWidth = densewidth)
# # C = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1500, uncertaintyEst = 'nlll', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.1, denseWidth = densewidth)

# # C = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1000, uncertaintyEst = 'nlll', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.05)
# # C = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1000, uncertaintyEst = 'nlll', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.01)

# # C = test_og_plot_models(training_generator, validation_generator, testsamples = ts, epochs = 1000, uncertaintyEst = 'nlll', LearningRate = 4e-3, fullDR = False, DropoutRate = 0.0)
# # print(f'{A = }')

# # mk_lr_epoch_tests()
# # mk_bs_epoch_tests()
# # mk_bs_lr_tests()


# plt.show()
"""
Prob drop multi params for now, prøv evt. at splitte netværket midt i (med functional approach), adjust learning rate, prøv at outputte kun en variabel selvom to varierer (for at guarantee den her så ændr måske data creator så NN bare tror der kun varieres en selvom der varieres to?), 
overvej ikke-diagonal led. sneaking suspicion at det måske er et mixup i plotting eftersom losses virker gode som counterpoint lader der dog til at være en fairly klar tendens til at følge predictions :/
Normalisering af træning & test data? (delta-felterne) - min max vil være nemt nok, Gaussian ret træls hvis alt træningsdata ikke fitter i memory
https://chatgpt.com/c/68316e9d-3498-8009-914e-db62407a8504

Tror apparently også at man kan lave to seperate netværk i en for flere output predictions? - ok ig det bare er at lave en class med to NN i sig https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

Loss med multiple outputs: https://keras.io/api/losses/
returnerer vist average loss for de forskellige parametre basically (sum_over_batch_size er default)

depthwise conv
https://www.reddit.com/r/MLQuestions/comments/gp2pj9/what_is_depthwiseconv2d_and_separableconv2d_in/

memory
https://stackoverflow.com/questions/46066850/understanding-the-resourceexhaustederror-oom-when-allocating-tensor-with-shape
https://medium.com/analytics-vidhya/reducing-deep-learning-model-size-without-effecting-its-original-performance-and-accuracy-with-a809b49cf519
https://www.reddit.com/r/tensorflow/comments/13vrs25/how_can_i_reduce_the_size_of_my_machine_learning/
https://www.tensorflow.org/tutorials/optimization/compression
https://www.omi.me/blogs/tensorflow-errors/out-of-memory-in-tensorflow-causes-and-how-to-fix

https://forums.fast.ai/t/training-4-5-million-parameters-takes-30-mins-per-epoch/83081/11





Prøv at træne bayesian med flere samples, evt. prøv andre priors

"""