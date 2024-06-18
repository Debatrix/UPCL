# %%
import matplotlib.pyplot as plt
import numpy as np
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
font = {'weight': 'bold', 'size': 11}
plt.grid(alpha=0.5)
plt.legend(ncol=2, loc=3, prop=font)

plt.savefig(f'../logs/{dataset}_{task}.pdf', bbox_inches='tight')
# %%
