import os
import random
from scipy.io.wavfile import read
import numpy as np
import pickle
import vad
import mel
import gmm
import time
from sys import maxint


database_path = '../data/yoho-enroll/'
mfcc_path = '../data/yoho-mfcc/'
test_path = '../data/yoho-verify/'
model_path = '../data/yoho-models/'

# FOR THE YOHO DATASET
#     161 enrollees, 385 utterances/enrollee
#     117 testers, 120 utterances/tester
# 
#     8000 samples/s sample frequency
#     29,440 samples/utterance

# 15 MFC-coefficients per frame
#
# Frame size = sample_frequency*0.025 = 200 samples
# Step size = sample_frequency*0.01 = 80 samples
# 182 frames per utterance
#
# Final MFCC per utterance: 15 x 182

# 32 Gaussian Mixtures per speaker


def scan_files():
    """ Scan the files located at ./Database/UserName/ and save a new .mfcc
    file, containing the MFCCs of each sample read.
    """

    num_users = len(os.listdir(database_path))
    for i, user in enumerate(os.listdir(database_path)):
        user_path = database_path + user + '/'
        print 'Analyzing files for user',user,i,'/',num_users
        num_wavs = len(os.listdir(user_path))
        for j, files in enumerate(os.listdir(user_path)):
            if files[-11:] == '.uncomp.wav':
                print '\tExtracting mfcc of: ' + user_path + files, j,'/',num_wavs
                mfcc = wav_to_mfcc(user_path + files)
                np.savetxt(user_path + files.split('.')[0] + '.mfcc', mfcc, newline='\n')


def save_all_mfcc():
    """ Scan all .mfcc files at ./Database/UserName/, group them together by
    username and save one .mfcc file for each user containing all mfcc for
    this user at mfcc_path
    """

    for user in os.listdir(database_path):
        user_path = database_path + user + '/'
        all_values = []
        for files in os.listdir(user_path):
            if files[-5:] == '.mfcc':
                mfcc = np.loadtxt(user_path + files)
                all_values.append(mfcc)

        for i in xrange(len(all_values)):
            if i == 0:
                mfcc = np.array(all_values[0])
            else:
                mfcc = np.append(mfcc, all_values[i], axis=0)

        np.savetxt(mfcc_path + user + '.mfcc', mfcc, newline='\n')

def load_mfcc(name):
    print 'Loading MFCC for',name
    path = mfcc_path + name + '.mfcc'
    mfcc = np.loadtxt(path)
    return mfcc
    

def train_model():
    """ Read all MFCC files at mfcc_path and use the coefficients to train
    models for each user. The model used is a Gaussian Mixture Model
    """

    models = []
    names = []
    for fyle in os.listdir(mfcc_path):
        if not (len(fyle) > 5) or fyle[-5:] != '.mfcc': continue
        name = fyle.split('.')[0]
        if model_exists(name): continue
        print 'Training ' + name + "'s model"
        mfcc = load_mfcc(name)
        model = gmm.gmm(mfcc)
        save_model(model, name)
        models.append(model)
        names.append(name)

    return models, names

SKLEARN_VERSION_FILEPATH = model_path + 'sklearn.version'

def model_exists(name):
    return name + '.model' in os.listdir(model_path)

def save_model(model, name):
    with open(SKLEARN_VERSION_FILEPATH, 'w') as f:
        f.write(gmm.get_version())

    path = model_path + name + '.model'
    print 'Saving', path
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def save_models(models, names):
    assert len(models) == len(names)

    with open(SKLEARN_VERSION_FILEPATH) as f:
        f.write(str(gmm.get_version()))

    for i, model in enumerate(models):
        name = names[i]
        path = model_path + name + '.model'
        save_model(model, name)

def assert_sklearn_version():
    with open(SKLEARN_VERSION_FILEPATH, 'r') as f:
        assert f.readline().strip() == gmm.get_version()

def load_models():
    models, names = [], []
    for filename in os.listdir(model_path):
        if '.model' in filename:
            name = filename[:-6]
            model = load_model(name)
            models.append(model)
            names.append(name)
    return models, names

def load_model(name):
    path = model_path + name + '.model'
    with open(path, 'rb') as f:
        model = pickle.load(f)
        return model

def wav_to_mfcc(path):
    data = read(input_path)
    print 'Extracting mfcc of: ' + test_path + file
    audio = data[1]
    f_sampling = data[0]
    audio_without_silence = vad.compute_vad(audio, f_sampling)
    mfcc = mel.extract_mfcc(audio_without_silence, f_sampling)
    return mfcc


def identify_speaker(models, names, input_path):
    """ Given the user models, returns the user that has the highest likelihood
    score with the input test passed in

    :param models: GMM models for each user in the database
    :param names: Name of each user

    :return: the index of the most likely user to be the one speaking in the
    input test and the mfcc from this input test.
    """

    mfcc = wav_to_mfcc(input_path)

    lk = -maxint - 1
    index = -1

    for i in xrange(len(models)):
        aux = gmm.get_likehood(models[i], mfcc)
        if aux > lk:
            lk = aux
            index = i

    return index, name

def test_identification(num_users = 10):
    assert num_users <= len(os.listdir(model_path)) - 1
    users = [fyle[:-6] for fyle in os.listdir(model_path) if fyle[-6:] == '.model']
    possible_test_users = [fyle for fyle in os.listdir(test_path) if len(fyle.split('.')) == 1]

    test_users = [user for user in users if user in possible_test_users][0:num_users]
    print 'Testing classification among enrolled users:',test_users

    models = []
    for user in test_users:
        path = model_path + user + '.model'
        with open(path, 'rb') as f:
            model = pickle.load(f)
        models.append(model)

    correct, incorrect = 0,0
    for user in test_users:
        print 'Testing for user', user
        i,c = 0,0
        for fyle in os.listdir(test_path + user):
            if fyle[-4:] == '.wav':
                print '\tTesting for user', user
                i, name = identify_speaker(models, test_users, test_path + user + '/' + fyle)
                if name != user: i += 1
                else: c += 1
        incorrect += i
        correct += c
        print 'Correct tallies:',c,'Incorrect tallies',i

    print 'total correct classifications:', correct
    print 'total incorrect classifications:', incorrect
    print 'accuracy', (correct + 0.0)/(correct+incorrect)

    return correct, incorrect

def verify_speaker(models, nameIndex, mfcc, threshold):
    """ Do the verification process to verify if the user really is the user chosen
    or is someone else from outside the database

    :param models: All models for each user in the database
    :param nameIndex: Index of the user chosen as the most probably
    :param mfcc: MFCC of the input test
    :param threshold: Threshold for the difference
    """

    total = 0

    for i in xrange(len(models)):
        if i != nameIndex:
            total += gmm.get_likehood(models[i], mfcc)

    total /= (len(models) - 1)

    candidate = gmm.get_likehood(models[nameIndex], mfcc)

    print "Difference between the probability of being the user chosen before or any other" \
          " user in the database is: " + str(candidate - total)

    if candidate - total > threshold:
        print "Most likely the user really is the one chosen"
    else:
        print "Probably the person who is speaking is from outside the database"

def build_ubm_mfcc(num_speakers, exclude):
    print 'Combining MFCCs for',num_speakers,'speakers'
    exclude.append('ubm')
    count = 0
    total_mfcc = None
    for fyle in os.listdir(mfcc_path):
        if fyle[-5:] == '.mfcc' and fyle[:-5] not in exclude:
            if count >= num_speakers: break
            else:
                print 'Adding MFCCs for UBM with speaker data from user',fyle[:-5]
                mfcc = np.loadtxt(mfcc_path + fyle)
                if total_mfcc is None:
                   total_mfcc = np.array(mfcc)
                else:
                    total_mfcc = np.append(total_mfcc, mfcc, axis=0)
                count += 1
    np.savetxt(mfcc_path + 'ubm.mfcc', total_mfcc, newline='\n')

def build_ubm_model(num_speakers, num_components, exclude=[]):
    print 'Building the UBM with number of speakers',num_speakers,'and number of components',num_components
    build_ubm_mfcc(num_speakers, exclude)
    mfcc = load_mfcc('ubm')
    model = gmm.gmm(mfcc, num_components=num_components)
    save_model(model, 'ubm')
    print 'Finished building and saving UBM'


def get_random_test_wav():
    random_user = random.choice(os.listdir(test_path))
    random_wav = random.choice(os.listdir(test_path + random_user))
    return test_path + random_user + '/' + random_wav, random_user


def test_verification(threshold = 250, test_speaker = '105', num_tests = 1000):
    if not model_exists('ubm'):
        ubm_model = build_ubm_model(num_speakers=10,num_components=200,exclude=[test_speaker])
    else:
        ubm_model = load_model('ubm')

    test_model = load_model(test_speaker)
    used_paths = set()

    print 'Running',num_tests,'verification tests with UBM against',test_speaker,'and threshold',threshold

    correct, incorrect = 0,0
    for i in range(num_tests):
        path, name = get_random_test_wav()
        while path in used_paths:
            path, name = get_random_test_wav()
        mfcc = wav_to_mfcc(path)

        test_likelihood = gmm.get_likelihood(test_model, mfcc)
        ubm_likelihood = gmm.get_likelihood(ubm_model, mfcc)
        
        print 'User and UBM likelihood',test_likelihood, ubm_likelihood

        if abs(test_likelihood - ubm_likelihood) > threshold and test_speaker == name:
            correct += 1
            print '\tTest',i,': correct'
        elif abs(test_likelihood - ubm_likelihood) <= threshold and test_speaker != name:
            correct += 1
            print '\tTest',i,': correct'
        else:
            incorrect += 1
            print '\tTest',i,': incorrect'

    print 'num speakers in UBM trained:', num_speakers
    print 'num components in UBM:', num_components
    print 'num tests:', num_tests
    print 'total correct classifications:', correct
    print 'total incorrect classifications:', incorrect
    print 'accuracy:', (correct + 0.0)/(correct + incorrect)

    return correct, incorrect

if __name__ == '__main__':
    start = time.time()
    print 'Starting the MFCC extraction'
    # scan_files()
    # save_all_mfcc()
    # models, names = train_model()
    # models, names = load_models()
    test_verification()
    # nameIndex, mfcc = identify_speaker(models, names)
    # verify_speaker(models, nameIndex, mfcc, 350)
    end = time.time()
    print str(end - start) + " seconds"
