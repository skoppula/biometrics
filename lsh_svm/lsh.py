
# coding: utf-8

# In[1]:

from svm_helpers import *


# In[2]:

def read_in_data():
    enr_ivec_path = "./processed_data/ivectors_enroll_500_600/all-enroll-ivectors.ark"
    (n_classes, n_features, n_ivectors, spk_labels, ivectors, utt_ids) = read_ivectors(enr_ivec_path)
    return (n_classes, n_features, n_ivectors, spk_labels, ivectors, utt_ids)
# (n_classes, n_features, n_ivectors, spk_labels, ivectors, utt_ids) = read_in_data()


# In[3]:

from random import gauss
from scipy.special import expit

def gen_hashes(ivectors, n_features, n_hashes, n_bits_per_hash, hyperplane_ratio=0.85):
    # Generate Cosine Hash hyperplants
    #   N_HASH_BLOCKS = M = L = 1
    #   Each Hash Fn: input = dimension n_features; output = # bits = dimension K = 10000
    #   total number of hyperplanes to produce for cosine hashes: K*M
    #   NOT implementing Moreno Lopez et al. (m choose 2) optimizations
    assert np.shape(ivectors)[1] == n_features
    print "generating hashes with",n_bits_per_hash,"bits per hash"
    
    n_hyperplanes = n_hashes*n_bits_per_hash
    n_cos_hp = hyperplane_ratio*n_hyperplanes

    hyperplanes = np.random.randn(n_features, n_hyperplanes)
    cos_hashes = (np.sign(np.dot(ivectors, hyperplanes[:,:n_cos_hp])) + 1)/2.0
    euc_hashes = np.floor(np.dot(ivectors, hyperplanes[:,n_cos_hp:]) % 2) # TODO CHANGE THIS BACK TO 2
    print np.shape(cos_hashes), np.shape(euc_hashes)
    hashes = (np.hstack((cos_hashes, euc_hashes))).astype(np.uint8)
    return hashes


# In[4]:

import math
from pyhashxx import hashxx

def identity_post_hash_fn(x): return x
def mod2_post_hash_fn(x): return x % 2

def get_salts(dim):
    return np.arange(dim) % 2

def apply_salt_and_hash(hashes, salts, post_hash_fn):
    nbph = np.shape(hashes)[1]
    size = str(int(math.ceil(math.log(nbph, 2)))+1)
    salted_str_hashes = (hashes + salts).astype('S'+size)
    def hxx(x): return post_hash_fn(hashxx(x))
    vect_hash_fn = np.vectorize(hxx)
    return vect_hash_fn(salted_str_hashes)

def apply_salt2_xor(hashes, salts):
    return np.bitwise_xor(hashes, salts)


# In[5]:

def compute_hamming_dists(X, vec):
    return np.sum(np.abs(X - vec), axis=1)

# n_cos_hyperplanes
def compute_ind_hamming_dists(X, vec, n_cos):
    hamming_dists = np.abs(X - vec)
    return np.sum(hamming_dists[:,:n_cos], axis=1), np.sum(hamming_dists[:,n_cos:], axis=1)

def get_top_n_guesses(hs_train, trn_labels, vec, n):
    # written to get top 'n' guesses, but we do not need that right now
    dists_to_vec = compute_hamming_dists(hs_train, vec)
    top_spk_idxs = np.argsort(dists_to_vec)
    top_guesses = [trn_labels[top_spk_idxs[i]] for i in range(n)]
    top_dists = [dists_to_vec[top_spk_idxs[i]] for i in range(n)]
    return top_guesses, top_guesses

def get_top_guess(hs_train, trn_labels, vec):
    # written to get top 'n' guesses, but we do not need that right now
    n = 1
    dists_to_vec = compute_hamming_dists(hs_train, vec)
    top_spk_idx = np.argmin(dists_to_vec)
    top_guess = trn_labels[top_spk_idx]
    top_dist = dists_to_vec[top_spk_idx]
    return top_guess, top_dist


# In[6]:

from collections import deque

def accuracy(n_correct, n_incorrect):
    return n_correct/(n_correct + n_incorrect)

def eval_approx_accuracy(hs_train, y_train, hs_test, y_test, stop_std_thres = 0.005, termination_window_size=200):
    print 'evaluating accuracy with stop std thres', stop_std_thres, ", frame size", termination_window_size
    n_correct, n_incorrect = 0.0,0.0
    accuracies = deque(range(termination_window_size))
    for i in range(len(y_test)):
        top_guess, top_dist = get_top_guess(hs_train, y_train, hs_test[i,])
        accuracies.popleft()
        if y_test[i] == top_guess:
            n_correct += 1 
        else:
            n_incorrect += 1
        acc = accuracy(n_correct, n_incorrect)
        old_acc = accuracies.popleft()
        if(np.std(accuracies) < stop_std_thres): break
        else: accuracies.append(acc)
    return acc, n_correct, n_incorrect

def eval_accuracy(hs_train, y_train, hs_test, y_test):
    print 'evaluating accuracy (full)'
    n_correct, n_incorrect = 0.0,0.0
    for i in range(len(y_test)):
        top_guess, top_dist = get_top_guess(hs_train, y_train, hs_test[i,])
        if y_test[i] == top_guess:
            n_correct += 1 
        else:
            n_incorrect += 1
    acc = accuracy(n_correct, n_incorrect)
    return acc, n_correct, n_incorrect

def plot_acc(nbphs, accs):
    plt.clf()
    fig = plt.figure()
    plt.plot(nbphs, accs, '-o')
    fig.suptitle('Accuracy as Function of LSH Size')
    plt.xlabel('Total # Bits used in LSHs')
    plt.ylabel('Test Set Accuracy')
    plt.show()


# In[7]:

accs = []
times = []
import time

def create_accuracy_per_nbphs_plot():
    global accs
    # values for number of bits per hash
    nbphs = range(3000,8200,200)[::-1]

    nbph = 8000
    print "testing num bits per hash:", nbph
    hashes = gen_hashes(ivectors, n_features, 1, nbph, hyperplane_ratio=0.9)
    salts = get_salts(nbph)
    new_hashes = apply_salt2_xor(hashes, salts)
    hs_train, hs_test, y_train, y_test = train_test_split(new_hashes, spk_labels)
    t0 = time.time()
    acc, n_correct, n_incorrect = eval_accuracy(hs_train, y_train, hs_test, y_test)
    times.append(time.time() - t0)
    print "got results:", acc, n_correct, n_incorrect
    accs.append(acc)
    print "\n"
        
    plot_acc(nbphs, accs)


# In[8]:

# create_accuracy_per_nbphs_plot()


# In[ ]:

def plot_dist_scores(hs_train, trn_labels, x, lbl, idx, nbph = 4000, num_spks = 50, save_dir = "hash_dist_viz/"):
    cos, euc = compute_ind_hamming_dists(hs_train, x, nbph)
    
    guessed_spk = trn_labels[np.argmin(cos + euc)]
    
    plt.clf()

    curr_spk = lbl
    chosen_spks = set(random.sample(set(y_train), num_spks))
    chosen_spks.add(curr_spk)

    plot_data = {}
    for chosen_spk in chosen_spks:
        plot_data[chosen_spk] = [[],[]]

    for i, spk in enumerate(y_train):
        if spk in chosen_spks:
            plot_data[spk][0].append(cos[i])
            plot_data[spk][1].append(euc[i])

    fig = plt.figure()
    fig.suptitle('Dist Between a Test Vector LSH and Each Train Vector LSH (Black = Target Spk)')
    plt.xlabel('Hamming Dist with Cosine LSH')
    plt.ylabel('Hamming Dist with Euc LSH')

    color_cycle=iter(cm.rainbow(np.linspace(0,1,len(chosen_spks)+1)))

    for spk in chosen_spks:
        nxt_color = 'k' if spk == curr_spk else next(color_cycle)
        plt.scatter(plot_data[spk][0], plot_data[spk][1], color=nxt_color)

    if guessed_spk == lbl:      
        if(save_dir):
            pylab.savefig(save_dir + str(idx) +".png",bbox_inches='tight')
        else:
            plt.show()
    else:
        if(save_dir):
            pylab.savefig(save_dir + "MISTAKE_" + str(idx) +".png",bbox_inches='tight')
        else:
            print "Mistake in classification!"
            plt.show()
            
def plot_all_dist_scores():
    for idx in range(len(y_test)):
        print "On idx", idx, "of", len(y_test)
        plot_dist_scores(hs_train, y_train, hs_test[idx,:], y_test[idx], idx)


# In[ ]:



