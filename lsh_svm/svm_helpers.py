
# coding: utf-8

# In[ ]:

import numpy as np

def file_len(fname):
    import subprocess

    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def read_ivectors(data_path):
    print("Reading in ivector data", data_path)
    '''
        takes in file like all-enroll-ivectors.ark with format:
           241-1-62_93_41  [ -0.905705 -1.320126 ... 0.1273378 -0.236616 ]
           241-1-63_46_67  [ -0.905705 -1.320126 ... 0.1273378 -0.236616 ]
            ...
        and returns:
           (n_classes, n_features, n_ivectors, spk_labels, ivectors, utt_ids)
           ivectors has format:
               [-0.905705 -1.320126 ... 0.1273378 -0.236616 ] -> utterance for spk 1
               [-0.905705 -1.320126 ... 0.1273378 -0.236616 ] -> utterance 2 for spk 1
               ...
               num of utterances
               ...
    '''
    # get the num_features and dimension of each ivector
    n_ivectors = file_len(data_path)
    with open(data_path, 'r') as f:
        first_line = f.readline().split("  ")
        ivector = np.fromstring(first_line[1].strip()[2:-2], dtype=float, sep=' ')
        n_features = len(ivector)

    ivector_count = 0
    ivectors = np.zeros(shape=(n_ivectors,n_features), dtype=float)
    spk_labels = np.zeros(shape=(n_ivectors,), dtype=int)
    utt_ids = np.zeros(shape=(n_ivectors,), dtype=str)
    truth_labels = []

    with open(data_path, "r") as f:
        for line in f:
            parts = line.split("  ")

            ivector = np.fromstring(parts[1].strip()[2:-2], dtype=float, sep=' ')
            assert len(ivector) == n_features
            ivectors[ivector_count] = ivector

            utt_name = parts[0]
            spk_labels[ivector_count] = int(utt_name.split("-")[0])
            utt_ids[ivector_count] = utt_name

            ivector_count += 1

    n_classes = len(set(spk_labels))
    print("number of distinct labels:", n_classes)
    print("number of features:", n_features)
    print("number of ivectors:", n_ivectors)
    print("i-vector matrix shape:", ivectors.shape)
    return (n_classes, n_features, n_ivectors, np.int64(spk_labels), ivectors, utt_ids)


# In[ ]:

import random
from matplotlib.pyplot import cm
import pylab

def plot_2dim_ivec(ivectors, spks, ivec_idxs = [0,1], save_dir="ivec_viz/"):
    spk_plot_data = {}
    for spk in spks:
        spk_plot_data[spk] = [[],[]]

    for i in range(len(ivectors)):
        if(spks[i] in spk_plot_data):
            for j in range(2):
                spk_plot_data[spks[i]][j].append(ivectors[i][ivec_idxs[j]])
    plt.clf()
    color=iter(cm.rainbow(np.linspace(0,1,len(spks))))
    for spk in spks:
        c=next(color)
        plt.scatter(spk_plot_data[spk][0], spk_plot_data[spk][1], color=c)
    if(save_dir):
        pylab.savefig(save_dir + "_".join([str(x) for x in ivec_idxs])+".png",bbox_inches='tight')
    else:
        plt.show()

def plot_all_dim_ivec(ivectors, spks, num_spks = 15, save_dir="ivec_viz/"):
    chosen_spks = random.sample(set(spks), num_spks)
    with open("ivec_viz/spks.txt", "w") as f:
        f.write(str(chosen_spks) + "\n")
    for i in range(np.shape(ivectors)[1]-1):
        plot_2dim_ivec(ivectors, spks, ivec_idxs=[i,i+1], save_dir=save_dir)


# In[ ]:

def spk_label_freqs(spk_labels):
    y = np.bincount(spk_labels)
    ii = np.nonzero(y)[0]
    return zip(ii,y[ii])


# In[ ]:

def binarize(spk_labels):
    # Binarize the output (one hot encoding of spk truth labels)
    from sklearn.preprocessing import label_binarize
    print("Binarizing labels...")
    bin_spk_labels = label_binarize(spk_labels, classes=list(set(spk_labels)))
    return bin_spk_labels

def train_test_split(ivectors, bin_spk_labels, test_size=0.1):
    from sklearn.model_selection import train_test_split
    print("Splitting into train and test. proportion: ", test_size)
    X_train, X_test, y_train, y_test = train_test_split(ivectors, bin_spk_labels, test_size=test_size, random_state=0)
    print("X train and test shapes", X_train.shape, X_test.shape, "y train and test shapes", y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


# In[ ]:

def train_svm(X_train, y_train):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import svm
    import numpy as np

    random_state = np.random.RandomState(0)
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    print("Training 1 v All SVM...")
    clf = classifier.fit(X_train, y_train)
    return clf

def compute_decision_function(clf, X_test):
    print('computing decision score (distance from hyperplane for each classifier)')
    y_score = clf.decision_function(X_test)
    return y_score


# In[ ]:

def compute_fpr_tpr_across_all_classes(y_test, y_score, n_classes):
    from sklearn.metrics import roc_curve, auc
    print("computing FPR and TPR and ROC AUC across all classes...")
    from scipy import interp
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Now calculate macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


# In[ ]:

import matplotlib.pyplot as plt

# nclass_index can either be one of the classes (1, n_classes) or 'micro' or 'macro'
def plot_idx_roc_curve(fpr, tpr, roc_auc, nclass_idx):
    plot_roc_curve(fpr[nclass_idx], tpr[nclass_idx], roc_auc[nclass_idx])

# nclass_index can either be one of the classes (1, n_classes) or 'micro'
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plot_dist_decision_boundary(y_score, y_test, idx):
    positive_ex = [e for i,e in enumerate(y_score[:,idx]) if y_test[:,idx] == 1]
    negative_ex = [e for i,e in enumerate(y_score[:,idx]) if y_test[:,idx] == 0]
    plt.clf()
    plt.scatter(positive_ex, [0]*len(positive_ex), color='b')
    plt.scatter(negative_ex, [0]*len(negative_ex), color='r')
    plt.show()


# In[ ]:

def compute_eer(fpr, tpr, nclass_idx):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    eer = brentq(lambda x : 1. - x - interp1d(fpr[nclass_idx], tpr[nclass_idx])(x), 0., 1.)
    print(eer)


# In[ ]:

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################


# In[ ]:

def main():
    enr_ivec_path = "./processed_data/ivectors_enroll_500_600/all-enroll-ivectors.ark"
    (n_classes, n_features, n_ivectors, spk_labels, ivectors, utt_ids) = read_ivectors(enr_ivec_path)
    bin_spk_labels = binarize(spk_labels)
    X_train, X_test, y_train, y_test = train_test_split(ivectors, bin_spk_labels)

    clf = train_svm(X_train, y_train)
    y_score = compute_decision_function(clf, X_test)
    fpr, tpr, roc_auc = compute_fpr_tpr_across_all_classes(y_test, y_score, n_classes)

    plot_idx_roc_curve(fpr, tpr, roc_auc, 'macro')
    compute_eer(fpr, tpr, 'macro')


# In[ ]:

# np.shape(y_score)


# In[ ]:



