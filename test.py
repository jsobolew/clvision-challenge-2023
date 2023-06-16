import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt

is_cuda = torch.cuda.is_available()

# test1 = np.load('pred_config_s3_run1_ewc0.2.npy')
# test2 = np.load('pred_config_s3_run1_ewc0.3.npy')
# test3 = np.load('pred_config_s3_run1_ewc0.4.npy')
# test4 = np.load('pred_config_s3_run1_ewc0.5.npy')
# test5 = np.load('pred_config_s3_run1_ewc0.6.npy')
# test6= np.load('pred_config_s3_run1_ewc0.7.npy')
# test7 = np.load('pred_config_s3_run1_ewc0.8.npy')

test1 = np.load('pred_config_s3_run1_lwf0.2.npy')
test2 = np.load('pred_config_s3_run1_lwf0.3.npy')
test3 = np.load('pred_config_s3_run1_lwf0.4.npy')
test4 = np.load('pred_config_s3_run1_lwf0.5.npy')
test5 = np.load('pred_config_s3_run1_lwf0.6.npy')
test6= np.load('pred_config_s3_run1_lwf0.7.npy')
test7 = np.load('pred_config_s3_run1_lwf0.8.npy')
gt= np.load('data/GT_test.npy')

lam = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
acc = []
for test in [test1, test2, test3, test4, test5, test6, test7]:
    acc.append(len(test[test==gt])/len(gt))

print(f'Mean: {np.mean(acc)}')

sns.pointplot(x=lam, y=acc)
plt.title('Accuracy depending on alpha parameter in LWF strategy')
plt.xlabel('Alpha value')
plt.ylabel('Evaluation accuracy')
plt.show()

print(matthews_corrcoef(test7, gt))
print(cohen_kappa_score(test7, gt))

testNAI = np.load('pred_config_s3_run1_lwf0.7.npy')
print(len(testNAI[testNAI==gt]))




