import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


results = np.load('results_config_s1_run1_NAI_alpha.npy', allow_pickle=True)
for metric in ['Top1_Acc_MB/train_phase/train_stream/Task000', 
                'Loss_MB/train_phase/train_stream/Task000', 
                'Top1_Acc_Epoch/train_phase/train_stream/Task000', 
                'Loss_Epoch/train_phase/train_stream/Task000', 
                'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000', 
                'Loss_Exp/eval_phase/test_stream/Task000/Exp000', 
                'Top1_Acc_Stream/eval_phase/test_stream/Task000', 
                'Loss_Stream/eval_phase/test_stream/Task000', 
                'StreamForgetting/eval_phase/test_stream']:
    Top1_Acc_MB = []

    for i in range(50):
        Top1_Acc_MB.append(results[i][metric])
    x = np.arange(0,50,1)
    sns.pointplot(x=x, y=Top1_Acc_MB)
    plt.title(metric)
    plt.xlabel('Experience')
    plt.ylabel('Evaluation loss')
    # plt.ylabel('Training loss')
    plt.show()






# dict_keys([
# 'Top1_Acc_MB/train_phase/train_stream/Task000', 
# 'Loss_MB/train_phase/train_stream/Task000', 
# 'Top1_Acc_Epoch/train_phase/train_stream/Task000', 
# 'Loss_Epoch/train_phase/train_stream/Task000', 
# 'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000', 
# 'Loss_Exp/eval_phase/test_stream/Task000/Exp000', 
# 'Top1_Acc_Stream/eval_phase/test_stream/Task000', 
# 'Loss_Stream/eval_phase/test_stream/Task000', 
# 'StreamForgetting/eval_phase/test_stream', 
# 'ConfusionMatrix_Stream/eval_phase/test_stream'])