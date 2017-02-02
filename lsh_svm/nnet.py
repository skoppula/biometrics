
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from svm_helpers import *
from lsh import *


# In[ ]:

enr_ivec_path = "./processed_data/ivectors_enroll_500_600/all-enroll-ivectors.ark"
(n_classes, n_features, n_ivectors, spk_labels, ivectors, utt_ids) = read_ivectors(enr_ivec_path)
nbph = 8000
hashes = gen_hashes(ivectors, n_features, 1, nbph, hyperplane_ratio=0.9)
salts = get_salts(nbph)


# In[ ]:

print salts[0:20]
print hashes[0,0:20]


# In[ ]:

new_hashes = apply_salt2_xor(hashes, salts)


# In[ ]:

bin_spk_labels = binarize(spk_labels)
trX, teX, trY, teY = train_test_split(new_hashes, bin_spk_labels)


# In[ ]:

import time

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, nbph])
Y = tf.placeholder("float", [None, 138])

w_h = init_weights([nbph, 625]) # create symbolic variables
w_o = init_weights([625, 138])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

accs = []
avg_times = []
num_test_spks = np.shape(teX)[0]
def evaluate_accuracy():
    t0 = time.time()
    np.bitwise_xor(hashes[0,:], salts)
    acc = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}))
    total_time_avg = (time.time()-t0)/(num_test_spks + 0.0)*1000
    avg_times.append(total_time_avg)
    accs.append(acc)
    return acc, total_time_avg


# In[ ]:

MAX_NUM_EPOCHS = 100
num_epochs = 0
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    old_acc = -1
    thres = 0.0001
    for i in range(MAX_NUM_EPOCHS):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        acc, tm = evaluate_accuracy()
        print(i, acc, tm)
        if acc - old_acc < thres:
            break
        else:
            num_epocs += 1
            old_acc = acc


# In[ ]:

for nbphs in range(3000,8200,200)[::-1]:
    


# In[ ]:

len(avg_times)
num_epochs = len(avg_times)


# In[ ]:

def plot_acc(x, accs):
    plt.clf()
    fig = plt.figure()
    plt.plot(x, accs, '-o')
    fig.suptitle('Accuracy as Function of 1-Layer Net Training Iterations (99.84% MAX)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()

def plot_times(x, times):
    plt.clf()
    fig = plt.figure()
    plt.plot(x, times, '-o')
    mx_time = str(max(times))
    fig.suptitle('Authentication Latency as Function of 1-Layer Net Training Iterations (%s MAX)' % mx_time)
    plt.xlabel('Iterations')
    plt.ylabel('Milliseconds')
    plt.show()
    
plot_acc(range(1,num_epochs+1), accs)
plot_times(range(1,num_epochs+1), avg_times)

