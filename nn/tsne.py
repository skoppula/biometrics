
# coding: utf-8

# In[3]:

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import verify_nnet
get_ipython().magic('matplotlib inline')


# In[ ]:

model = TSNE(n_components=2, perplexity=50, random_state=1)
spk = "101"


# In[ ]:

out = model.fit_transform(pl_act)


# In[ ]:

plt.title("BH-SNE Embedding of Last Layer ReLU activations")
plt.scatter(out[:,0], out[:,1], c=pl_te, cmap=plt.cm.get_cmap("prism", 2))
plt.colorbar(ticks=range(1,3))
plt.show()


# In[4]:

N_INP_FRMS = 120

print("\nNUM INPUT FRAMES:",N_INP_FRMS,"\n")
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


# In[15]:

enr_spks = set(enr_y)
ver_spks = set(ver_y)

print(len(enr_spks), len(ver_spks))
print(min(enr_spks), max(enr_spks), max(enr_spks)-min(enr_spks)+1)
print(enr_spks - ver_spks)
print(ver_spks - enr_spks)


# In[16]:

spk_mappings = {}
curr_map = 0
for spk in enr_spks.union(ver_spks):
    if spk not in spk_mappings:
        spk_mappings[spk] = curr_map
        curr_map += 1
map_spks = np.vectorize(lambda x: spk_mappings[x])
enr_y = map_spks(enr_y)
ver_y = map_spks(ver_y)


# In[20]:

enr_spks = set(enr_y)
ver_spks = set(ver_y)

print(len(enr_spks), len(ver_spks))
print(min(enr_spks), max(enr_spks), max(enr_spks)-min(enr_spks)+1)
print(enr_spks - ver_spks)
print(ver_spks - enr_spks)

