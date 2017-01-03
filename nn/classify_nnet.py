# coding: utf-8

import numpy as np
import os
import gc

# all utterance data [utt1 mfcc frames [N1 x 60], utt2 frames]
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_verify_lbls(y, curr_spk):
    return np.array(list(map(lambda x: 1 if x == curr_spk else 0, y)))

def conv_to_ver_and_one_hot_encode(y, curr_spk):
    # convert to verification task
    ver_lbls = get_verify_lbls(y, curr_spk)
    num_frames = np.shape(y)[0]

    one_hot_lbls = np.zeros((num_frames, 2))
    one_hot_lbls[np.arange(num_frames), ver_lbls] = 1
    return one_hot_lbls

def get_ver_network_arch(model_path, n_inp_frms):

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.utils.visualize_util import plot
    from keras.optimizers import Adam

    # Add batch normalization: keras.layers.normalization.BatchNormalization()
    model = Sequential()
    model.add(Dense(128, input_shape=(n_inp_frms*60,))) # 60 MFCCs / frame
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.summary()
    plot(model, to_file=model_path + 'architecture.png')

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model

def evaluate_activations(model, X, layer):
    from keras.layers.core import K
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    return get_layer_output([X, 0])[0]

def train_and_test_network(model, tr_x, tr_y, te_x, te_y, curr_spk, model_path, n_epochs=75, batch_size=50):

    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, EarlyStopping

    saved_model_path = model_path + "curr_best_weights_"  + str(curr_spk) + ".hdf5"
    ckpt = ModelCheckpoint(saved_model_path, monitor='val_acc', verbose=0, save_best_only=False, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    trn_history = model.fit(tr_x, tr_y, validation_split=0.2,
                        batch_size=batch_size, nb_epoch=n_epochs, verbose=1,
                        callbacks=[ckpt, early_stop])

    np.save(model_path + "history_" + str(curr_spk) + ".npy", trn_history.history)

    model.load_weights(saved_model_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # del tr_x, tr_y, val_x, val_y # for saving memory
    score = model.evaluate(te_x, te_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    gc.collect() # fix suggested by https://github.com/tensorflow/tensorflow/issues/3388

    model.load_weights(saved_model_path)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    activations = evaluate_activations(model, te_x, 9)
    np.save(model_path + "activations_" + str(curr_spk) + ".npy", activations)

    predictions = model.predict(te_x, batch_size=batch_size, verbose=0)
    np.save(model_path + "predictions_" + str(curr_spk) + ".npy", predictions)

    return score[1]


if __name__ == "__main__":
    np.random.seed(1337) # reproducibility

    for N_INP_FRMS in [60, 80, 120]:
        print("\nNUM INPUT FRAMES:",N_INP_FRMS,"\n")
        MODEL_PATH = 'tsne_model_' + str(N_INP_FRMS) + '/'
        mkdir(MODEL_PATH)
        BASE_PATH = "/home/skoppula/biometrics/data/yoho/kaldi_yoho/data/"
        VER_PATH = BASE_PATH + "verify/final/nn_inp-" + str(N_INP_FRMS) + "_frames/"
        ENR_PATH = BASE_PATH + "enroll/final/nn_inp-" + str(N_INP_FRMS) + "_frames/"

        print("Model(s) Path: ", MODEL_PATH)
        print("Using verification path: ", VER_PATH)
        print("Using enroll data path: ", ENR_PATH)

        # ENROLL DATA LOAD
        enr_x = np.load(ENR_PATH + "X.npy")
        n_frames = np.shape(enr_x)[0]
        enr_x = enr_x.reshape(n_frames, N_INP_FRMS * 60) # 60 MFCCs per frame
        enr_y = np.load(ENR_PATH + "y.npy")
        assert n_frames == np.shape(enr_y)[0]
        print("Enroll X shape", np.shape(enr_x))
        print("Enroll y shape", np.shape(enr_y))

        # VERIFY DATA LOAD
        ver_x = np.load(VER_PATH + "X.npy")
        n_frames = np.shape(ver_x)[0]
        ver_x = ver_x.reshape(n_frames, N_INP_FRMS * 60)
        ver_y = np.load(VER_PATH + "y.npy")
        assert n_frames == np.shape(ver_y)[0]
        print("Verify X shape", np.shape(ver_x))
        print("Verify y shape", np.shape(ver_y))

        poss_spks = list(set(np.load(VER_PATH + "y.npy")))

        test_accs = []
        for i, curr_spk in enumerate(poss_spks):
            print("EVALUATING AND TRAINING FOR CURRENT SPEAKER:", curr_spk, str(i) + "/" + str(len(poss_spks)))
            model = get_ver_network_arch(MODEL_PATH, N_INP_FRMS)

            # oversampling data augmentation to counteract imbalanced dataset
            OVERSAMPLING_FACTOR = 60
            pos_idxs = np.where(enr_y == curr_spk)
            enr_y = np.concatenate((np.repeat(enr_y[pos_idxs], OVERSAMPLING_FACTOR), enr_y))
            enr_x = np.concatenate((np.repeat(enr_x[pos_idxs], OVERSAMPLING_FACTOR, axis=0), enr_x), axis=0)

            tr_y = conv_to_ver_and_one_hot_encode(enr_y, curr_spk)
            te_y = conv_to_ver_and_one_hot_encode(ver_y, curr_spk)
            spk_path = MODEL_PATH + str(curr_spk) + "/"
            mkdir(spk_path)
            test_acc = train_and_test_network(model, enr_x, tr_y, ver_x, te_y, curr_spk, spk_path)
            test_accs.append(test_acc)

        gc.collect()
        print("TEST ACCURACIES:",test_accs)
        print("Final average test accuracy:", sum(test_accs)/len(test_accs))

