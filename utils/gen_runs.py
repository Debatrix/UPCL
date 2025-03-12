# %%
import os
import json
import os.path as osp

os.chdir('..')
# %%
model = [
    'upcl',
]
inc = [5,10,20]
dataset = ['cifar100',]

device = ["3"]
seed = [2024, 2023, 1993]
mem = 2000
# %%
cmd = ''
for m in model:
    if not osp.exists(osp.join('exps', m + '.json')):
        print(osp.join('exps', m + '.json') + ' not exist!')
        continue
    with open(osp.join('exps', m + '.json'), 'r') as f:
        exp = json.load(f)

    exp['memory_size'] = mem
    exp['prefix'] = f'fixed'
    exp['device'] = device
    exp['seed'] = seed
    exp['lt_imb_factor'] = 0
    for d in dataset:
        exp['dataset'] = d
        for i in inc:
            exp['increment'] = i
            exp['init_cls'] = 50

            name = f'fixed_{m}_{d}_B0I{i}_M{mem}.json'
            # if osp.exists(osp.join('logs', m, d, f'0_{i}')):
            #     print(f'{name} exist, pass!')
            #     continue

            with open(f'runs/{name}', 'w') as f:
                json.dump(exp, f)
            cmd += f'python main.py --config=runs/{name};'

with open('runs/0_cmd.txt', 'a+') as f:
    f.write(cmd + '\n\n')

# %%
