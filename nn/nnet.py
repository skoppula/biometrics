
# coding: utf-8

# In[2]:

from sklearn.preprocessing import label_binarize
import numpy as np
import random
import os

np.random.seed(1337) # reproducibility

def binarize(spk_labels):
    # Binarize the output (one hot encoding of spk truth labels)
    print("Binarizing labels...")
    bin_spk_labels = label_binarize(spk_labels, classes=list(set(spk_labels)))
    return bin_spk_labels

# because sklearn 0.17 no longer has sklearn.model_selection.train_test_split
# NO LONGER USED BECAUSE KERAS MODEL FIT DOES VALIDATION SPLITS FOR YOU
def dataset_split(X, y, test_size=0.1):
    num_els = np.shape(X)[0]
    assert num_els == np.shape(y)[0]
    assert test_size <= 1 and test_size >= 0
    
    print("Splitting into train/test. Test proportion:", test_size)
    idxs = np.random.choice(np.arange(num_els), int((1-test_size)*num_els), replace=False)
    inv_idxs = list(set(np.arange(num_els)) - set(idxs))
    X_train, y_train, X_test, y_test = X[idxs], y[idxs], X[inv_idxs], y[inv_idxs]
    
    print("X,y train shapes", X_train.shape, y_train.shape, "X,y test shapes", X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test

 # all utterance data [utt1 mfcc frames [N1 x 60], utt2 frames]
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[3]:

N_INP_FRMS = 120
MODEL_PATH = 'model_' + str(N_INP_FRMS) + '/'
mkdir(MODEL_PATH)
BASE_PATH = "/home/skoppula/biometrics/data/yoho/kaldi_yoho/data/"
VER_PATH = BASE_PATH + "verify/final/nn_inp-" + str(N_INP_FRMS) + "_frames/"
ENR_PATH = BASE_PATH + "enroll/final/nn_inp-" + str(N_INP_FRMS) + "_frames/"


ENR_X = np.load(ENR_PATH + "X.npy")
ENR_y = np.load(ENR_PATH + "y.npy")

NUM_FRAMES = np.shape(ENR_X)[0]
ENR_X = ENR_X.reshape(NUM_FRAMES,N_INP_FRMS * 60) # 60 MFCCs per frame
assert NUM_FRAMES == np.shape(ENR_y)[0]

print("Enroll X shape", np.shape(ENR_X))
print("Enroll y shape", np.shape(ENR_y))

poss_spks = np.load(VER_PATH + "y.npy")

curr_spk = 109
model = get_ver_network_arch()
tr_x, tr_y = ENR_X, conv_to_ver_and_one_hot_encode(ENR_y, curr_spk)


# need spk in verify set (for now) to test positive authentication
# VER_y_60 = np.load("/home/skoppula/biometrics/data/yoho/kaldi_yoho/data/verify/final/nn_inp-60_frames/y.npy")
# curr_spk = random.choice(ENR_y_60)
# while curr_spk not in VER_y_60:
#   curr_spk = print(random.choice(ENR_y_60))


# In[ ]:

def get_verify_lbls(y, curr_spk):
    return np.array(list(map(lambda x: 1 if x == curr_spk else 0, y)))

def conv_to_ver_and_one_hot_encode(y, spk):
    # convert to verification task
    ver_lbls = get_verify_lbls(y, curr_spk)
    num_frames = np.shape(y)[0]
    
    one_hot_lbls = np.zeros((num_frames, 2))
    one_hot_lbls[np.arange(num_frames), ver_lbls] = 1
    return one_hot_lbls


# In[4]:

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot
from keras.layers.core import K

def get_ver_network_arch():
    # Add batch normalization: keras.layers.normalization.BatchNormalization()
    model = Sequential()
    model.add(Dense(64, input_shape=(N_INP_FRMS*60,))) # 60 MFCCs / frame
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.summary()
    plot(model, to_file=MODEL_PATH + 'architecture.png')
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy', 'precision', 'recall'])
    return model

def evaluate_activations(model, X, layer):
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    return get_layer_output([X, 0])[0]

def train_and_test_network(model, tr_x, tr_y, curr_spk, MODEL_PATH):
    
    EPOCHS = 2
    BATCH_SIZE = 50
    
    saved_model_path = MODEL_PATH + str(curr_spk) + "_curr_best_weights.hdf5"
    ckpt = ModelCheckpoint(saved_model_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    trn_history = model.fit(tr_x, tr_y, validation_split=0.2,
                        batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1, 
                        callbacks=[ckpt])
    
    model.load_weights(saved_model_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy', 'precision', 'recall'])

    te_x = np.load(VER_PATH + "X.npy")
    NUM_TEST_FRAMES = np.shape(te_x)[0]
    te_x = te_x.reshape(NUM_TEST_FRAMES,N_INP_FRMS * 60)
    te_y = conv_to_ver_and_one_hot_encode(np.load(VER_PATH + "y.npy"), curr_spk)
    assert NUM_TEST_FRAMES == np.shape(te_y)[0]

    print("Verify X shape", np.shape(te_x))
    print("Verify y shape", np.shape(te_y))
    
    # del tr_x, tr_y, val_x, val_y # for saving memory
    score = model.evaluate(te_x, te_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    activations = get_activations(model, te_x, 2)
    np.save(MODEL_PATH + "activations_" + str(curr_spk) + ".npy", activations)
    np.save(MODEL_PATH + "history_" + str(curr_spk) + ".npy", trn_history.history)


# In[ ]:




# In[ ]:




# In[ ]:




# In[8]:




# In[9]:




# In[ ]:



