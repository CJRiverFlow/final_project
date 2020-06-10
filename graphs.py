"""
File to print the model comparision graphs from stats csv files

NOTES: To do fair comparisions, run the current script after testing
       the 3 different model precision with the recorded video.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_int8 = pd.read_csv("stats_INT8.csv", index_col=0)
df_fp16 = pd.read_csv("stats_FP16.csv",index_col=0)
df_fp32 = pd.read_csv("stats_FP32.csv",index_col=0)

#LOADING TIME

labels = df_int8.columns.values
load_int =  [int(i) for i in df_int8.iloc[0,:]*1000]
load_fp16 = [int(i) for i in df_fp16.iloc[0,:]*1000]
load_fp32 = [int(i) for i in df_fp32.iloc[0,:]*1000]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(2)
rects1 = ax[0].bar(x - width, load_int, width, label='int8')
rects2 = ax[0].bar(x, load_fp16, width, label='fp16')
rects3 = ax[0].bar(x + width, load_fp32, width, label='fp32')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Time (ms)')
ax[0].set_title('Loading time by model and precision')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].legend()


def autolabel(rects, axis):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        axis.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1, ax[0])
autolabel(rects2, ax[0])
autolabel(rects3, ax[0])

#INFERENCE

infer_int =  [int(i) for i in df_int8.iloc[1,:]*1000]
infer_fp16 = [int(i) for i in df_fp16.iloc[1,:]*1000]
infer_fp32 = [int(i) for i in df_fp32.iloc[1,:]*1000]

rects4 = ax[1].bar(x - width, infer_int, width, label='int8')
rects5 = ax[1].bar(x, infer_fp16, width, label='fp16')
rects6 = ax[1].bar(x + width, infer_fp32, width, label='fp32')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Time (ms)')
ax[1].set_title('Inference time by model and precision')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].legend()

autolabel(rects4,ax[1])
autolabel(rects5,ax[1])
autolabel(rects6,ax[1])

#PLOTING
fig.tight_layout()

plt.show()


