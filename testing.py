import os
import matplotlib.pyplot as plt
from model import read_mfcc
from sklearn.metrics import roc_curve, auc
from debug import dpr
from debug import pr

# CLASSES:
#   0 = POSER
#   1 = NON-POSER (true enrollee)
NUM_CLASSES = 2

YOHO_PATH = '../data/yoho-mfcc'

'''
    Runs tests using data from a specific dataset

    Optional input args:
        num_tests_0   number of tests to try with true assignment class 0
        num_tests_1   number of tests to try with true assignment class 1
        dataset       one of: 'yoho', 'timit', 'nist'
    Returns tuple of three values:
        NUM_TESTS   the number of tests in this dataset
        assignments the true class assignment of this test
                    (0=poser/1=non-poser)
        scores      the score output by running the test data
                    on the authenticator/classifier
'''
# TODO Support the NIST and TIMIT datasets
def run_tests(num_tests_0=100, num_tests_1=100,dataset='yoho'):
    if dataset is 'yoho':
        pr('Testing with YOHO dataset with ' + str(num_tests_0)
                + 'class 0 tests and ' + str(num_tests_1) + ' class 1 tests'
                + 'with files at' + YOHO_PATH)
        
        users_paths = os.listdir(test_path)


        return NUM_TESTS, assignments, y_score
    elif dataset is 'timit':
        pass
    elif dataset is 'nist':
        pass
    pass

'''
    Generates ROC and DET curves, computes Area
    Under Curve (AUC) for ROC curve, and Equal
    Error Rate (EER) 

    input args:
        assignments     the true class assignment for each test
        scores          the classifier score for each test
        filename_base   the path/filename base where all output
                        is saved with given name base

    output:
        Nothing returned, image and summary statistics
        saved as *.png and *.txt files with input
        filename base

'''
# TODO generate EER and det curves as well
# Code from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def interpret_results(assignments, scores, filename_base="results/"):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


##############################################################################
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

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

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                         label='micro-average ROC curve (area = {0:0.2f})'
                                        ''.format(roc_auc["micro"]),
                                                 linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                         label='macro-average ROC curve (area = {0:0.2f})'
                                        ''.format(roc_auc["macro"]),
                                                 linewidth=2)

        for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                                           ''.format(i, roc_auc[i]))

                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Some extension of Receiver operating characteristic to multi-class')
                plt.legend(loc="lower right")
                plt.show()
