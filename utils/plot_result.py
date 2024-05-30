# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import json
from glob import glob
# %%
method = ['iCaRL', 'WA', 'NCCIL', 'UPCL']
dataset = 'imagenet100'
task = '0_20'
memory = ['M500', 'M1000', 'M2000']
marker = ["o", "s", "D"]
color = ['tomato', 'orange', 'green', 'blueviolet']

data = {}
for name in method:
    for mem in memory:
        rpath = f'../logs/{name.lower()}/{dataset}/{task}/*{mem}*.json'
        path = glob(rpath)
        if mem == 'M2000' and len(path) == 0:
            path = glob(
                f'../logs/{name.lower()}/{dataset}/{task}/reproduce_1993_resnet18.json'
            )
        assert len(path) == 1, rpath
        with open(path[0], 'r') as f:
            content = f.read()
            content = content.replace('\n', '').replace(' ', '')
            content = content.split('}{')
            d = json.loads('{' + content[-1])
            data[f'{name}_{mem}'] = {
                'cnn_top1_curve': d['cnn_top1_curve'],
                'cnn_avg_acc': d['cnn_avg_acc']
            }

plt.figure(figsize=(6, 6))
x = int(task.split('_')[-1])
x = np.array([a for a in range(x, 101, x)]).astype(int)
# x = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
for nn, name in enumerate(method):
    for nm, mem in enumerate(memory):
        plt.plot(
            x,
            data[f'{name}_{mem}']['cnn_top1_curve'],
            marker=marker[nm],
            color=color[nn],
            alpha=0.75,
            label=f'{name}_{mem}',
        )
# plt.xticks(x)
font = {'weight': 'bold', 'size': 11}
plt.grid(alpha=0.5)
plt.legend(ncol=2, loc=3, prop=font)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.xticks(fontproperties='Times New Roman', size=14)

# handles = []
# labels = []
# for nn, name in enumerate(method):
#     label = name + '('
#     for nm, mem in enumerate(memory):
#         label = '{}{:.2f}%,'.format(label,
#                                     data[f'{name}_{mem}']['cnn_avg_acc'])
#     handles.append(mpatches.Patch(color=color[nn]))
#     labels.append(label[:-1] + ')')
# plt.legend(handles=handles, labels=labels, loc=1)
plt.savefig(f'../logs/{dataset}_{task}.pdf', bbox_inches='tight')
# %%
