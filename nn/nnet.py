
# coding: utf-8

# In[ ]:

from sklearn.preprocessing import label_binarize

ENR_X_60 = np.load("/home/skoppula/biometrics")

def binarize(spk_labels):
    # Binarize the output (one hot encoding of spk truth labels)
    print "Binarizing labels..."
    bin_spk_labels = label_binarize(spk_labels, classes=list(set(spk_labels)))
    return bin_spk_labels

