import trn_classify
import numpy as np
import gc


def evaluate_activations(model, X, layer):
    from keras.layers.core import K
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    return get_layer_output([X, 0])[0]

if __name__ == "__main__":

    print("Starting testing script...")
    params = trn_classify.get_params()
    N_INP_FRMS, BASE_PATH, MODEL_PATH, VER_PATH, ENR_PATH = params
    print("PARAMS: ", params)

    _, enr_y = trn_classify.read_data(ENR_PATH, N_INP_FRMS)
    ver_x, ver_y = trn_classify.read_data(VER_PATH, N_INP_FRMS)
    all_spks, _, ver_y, spk_mappings = trn_classify.remap_spk_ids(enr_y, ver_y)
    ver_y = trn_classify.one_hot_encode(ver_y, len(all_spks))

    saved_model_path = MODEL_PATH + "curr_best_weights.hdf5"
    model = trn_classify.get_class_net(MODEL_PATH, N_INP_FRMS, len(all_spks))
    model.load_weights(saved_model_path)

    print("Shuffling dataset!")
    ver_x, ver_y = trn_classify.dataset_shuffle(ver_x, ver_y)

    print("Evaluating on the testing dataset!")
    score = model.evaluate(ver_x, ver_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    gc.collect()
    model2 = trn_classify.get_class_net(MODEL_PATH, N_INP_FRMS, len(all_spks))
    model2.load_weights(saved_model_path)
    activations = evaluate_activations(model2, ver_x, 7)
    np.save(MODEL_PATH + "activations.npy", activations)
    print("Done saving activations!")
