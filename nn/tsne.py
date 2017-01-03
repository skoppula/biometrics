
# coding: utf-8

# In[44]:

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import verify_nnet
get_ipython().magic('matplotlib inline')


# In[63]:

model = TSNE(n_components=2, perplexity=50, random_state=1)
spk = "101"


# In[68]:

activations = np.load("tsne_model_60/" + spk + "/activations_" + spk + ".npy")
activations.shape


# In[ ]:




# In[95]:

BASE_PATH = "/home/skoppula/biometrics/data/yoho/kaldi_yoho/data/"
VER_PATH = BASE_PATH + "verify/final/nn_inp-60_frames/"
lbls = np.load(VER_PATH + "y.npy")
te_x = np.load(VER_PATH + "X.npy")
te_y = verify_nnet.get_verify_lbls(lbls, int(spk))
te_y.shape


# In[97]:

pidxs = np.where(te_y[0:300]==1)[0]


# In[108]:

te_x.shape


# In[110]:

np.concatenate((np.repeat(lbls[pidxs], 2), lbls))
np.concatenate((np.repeat(te_x[pidxs], 10, axis=0), te_x), axis=0).shape


# In[112]:




# In[89]:

y = np.bincount(lbls)
ii = np.nonzero(y)[0]
list(zip(ii,y[ii])), len(lbls)


# In[86]:

preds = np.load("tsne_model_60/" + spk + "/predictions_" + spk + ".npy")
preds.shape
diffs = np.where(preds[:,0] < preds[:,1])


# In[87]:

pl_act = activations[0:300]
pl_te = te_y[0:300]
pl_pred = preds[0:300]
# print(sum(te_y))
# print(te_y[104:112])
print(lbls[104:112])
print(pl_act[104:112])
print(pl_pred[104:112])
# print(pl_act.shape, pl_te.shape)
preds[diffs], activations[diffs]


# In[53]:

out = model.fit_transform(pl_act)


# In[54]:

plt.title("BH-SNE Embedding of Last Layer ReLU activations")
plt.scatter(out[:,0], out[:,1], c=pl_te, cmap=plt.cm.get_cmap("prism", 2))
plt.colorbar(ticks=range(1,3))
plt.show()

