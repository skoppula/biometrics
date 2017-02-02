import numpy as np
import os
# np.random.seed(17) # reproducibility

# all utterance data [utt1 mfcc frames [N1 x 60], utt2 frames]
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# assumes inputs in y from 0 ... n
def one_hot_encode(y, num_spks):
    num_frames = np.shape(y)[0]
    one_hot_lbls = np.zeros((num_frames, num_spks))
    one_hot_lbls[np.arange(num_frames), y] = 1
    return one_hot_lbls

def get_class_net(model_path, n_inp_frms, num_spks):

    from keras.models import Sequential
    from keras.layers.core import MaxoutDense, Dense, Dropout, Activation
    from keras.layers.advanced_activations import LeakyReLU
    from keras.utils.visualize_util import plot
    from keras.optimizers import Adam

    # Add batch normalization: keras.layers.normalization.BatchNormalization()
    model = Sequential()
    model.add(Dense(512, input_shape=(n_inp_frms*60,))) # 60 MFCCs / frame
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(MaxoutDense(128))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Dense(num_spks))
    model.add(Activation('softmax'))

    model.summary()
    plot(model, to_file=model_path + 'architecture.png')

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy', 'precision', 'recall'])
    return model

def train_network(model, tr_x, tr_y, model_path, n_epochs=150, batch_size=50):

    from keras.callbacks import ModelCheckpoint

    saved_model_path = model_path + "curr_best_weights.hdf5"
    ckpt = ModelCheckpoint(saved_model_path, save_best_only=False)
    # early_stop = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    trn_history = model.fit(tr_x, tr_y, validation_split=0.2,
                        batch_size=batch_size, nb_epoch=n_epochs, verbose=1,
                        callbacks=[ckpt])

    np.save(model_path + "history.npy", trn_history.history)

def remap_spk_ids(enr_y, ver_y):
    # REMAP SPEAKER IDs TO 0..n
    spk_mappings = {}
    curr_map = 0
    all_spks = set(enr_y).union(set(ver_y))
    for spk in all_spks:
        if spk not in spk_mappings:
            spk_mappings[spk] = curr_map
            curr_map += 1
    map_spks = np.vectorize(lambda x: spk_mappings[x])
    print("Speaker Re-mappings:", spk_mappings)
    return all_spks, map_spks(enr_y), map_spks(ver_y), spk_mappings

def read_data(path, n_inp_frms):
    x = np.load(path + "X.npy")
    n_frames = np.shape(x)[0]
    x = x.reshape(n_frames, n_inp_frms * 60) # 60 MFCCs per frame
    y = np.load(path + "y.npy")
    assert n_frames == np.shape(y)[0]
    return x, y

def get_params():
    N_INP_FRMS = 120
    BASE_PATH = "/home/skoppula/biometrics/data/yoho/kaldi_yoho/data/"
    MODEL_PATH = 'classify_model_' + str(N_INP_FRMS) + '/'
    mkdir(MODEL_PATH)
    VER_PATH = BASE_PATH + "verify/final/nn_inp-" + str(N_INP_FRMS) + "_frames/"
    ENR_PATH = BASE_PATH + "enroll/final/nn_inp-" + str(N_INP_FRMS) + "_frames/"

    return N_INP_FRMS, BASE_PATH, MODEL_PATH, VER_PATH, ENR_PATH


def dataset_shuffle(X, y):
    num_els = np.shape(X)[0]
    assert num_els == np.shape(y)[0]
    idxs = np.random.permutation(np.arange(num_els))
    return X[idxs], y[idxs]


if __name__ == '__main__':

    params = get_params()
    N_INP_FRMS, BASE_PATH, MODEL_PATH, VER_PATH, ENR_PATH = params
    print("PARAMS: ", params)

    enr_x, enr_y = read_data(ENR_PATH, N_INP_FRMS)
    _, ver_y = read_data(VER_PATH, N_INP_FRMS)
    del _

    print("Enroll X shape", np.shape(enr_x))
    print("Enroll y shape", np.shape(enr_y))
    print("Verify y shape", np.shape(ver_y))

    all_spks, enr_y, _, _ = remap_spk_ids(enr_y, ver_y)
    model = get_class_net(MODEL_PATH, N_INP_FRMS, len(all_spks))

    enr_y = one_hot_encode(enr_y, len(all_spks))
    print("One-Hot Enroll (y) shape", np.shape(enr_y))
    del ver_y, all_spks

    print("Shuffling dataset!")
    enr_x, enr_y = dataset_shuffle(enr_x, enr_y)

    train_network(model, enr_x, enr_y, MODEL_PATH)
    print("Training done!")

