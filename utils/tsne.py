# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from  matplotlib.lines import Line2D
import matplotlib.colors as mcolors
# %%
baseline = torch.load('../logs/baseline.pth')
upcl = torch.load('../logs/upcl.pth')
# %%
def similar_colors(hex_color, num):
    rgb_color = mcolors.hex2color(hex_color)
    hsv_color = mcolors.rgb_to_hsv(rgb_color)
    colors = []
    for offset in np.linspace(-0.1, 0.1, num):
        # offset = i / num / 2  # 色调偏移量
        new_hue = (hsv_color[0] + offset) % 1  # 新的色调
        new_color = mcolors.hsv_to_rgb([new_hue, hsv_color[1], hsv_color[2]])
        colors.append(mcolors.rgb2hex(new_color))
    return colors
# %%
def project(tensor, type='tsne'):
    if type == 'tsne':
        projector = TSNE(
            n_components=2,
            init='pca',
            learning_rate='auto',
            n_iter=3000,
            perplexity=25,
            verbose=True,
        )
    else:
        projector = PCA(n_components=2)
    return projector.fit_transform(tensor)

b_feat = baseline['features'].reshape(10,10,100,-1)
b_proto = baseline['prototype'].reshape(10,10,-1)

b_all_feat = np.concatenate((b_feat[0].reshape(-1,512),b_feat[-1].reshape(-1,512),b_proto[0],b_proto[-1]),0)

b_d_feat = project(b_all_feat.astype(np.double), 'tsne')

b_feat = b_d_feat[:2000].reshape(2,10,100,-1)
b_proto = b_d_feat[2000:].reshape(2,10,-1)


plt.figure()
colors = ['#65{:02x}08'.format(i) for i in range(100, 200, 10)]
for i in range(10):
    plt.scatter(b_feat[0,i,:,0],
                    b_feat[0,i,:,1],
                    s=20,
                    color=colors[i],
                    marker='o',
                    label='old classes',
                    edgecolor='none',
                    alpha=0.5)
    plt.scatter(b_proto[0,i,0],
                    b_proto[0,i,1],
                    s=20,
                    color='r',
                    marker='x',
                    label='prototype',
                    alpha=1)

colors = ['#0165{:02x}'.format(i) for i in range(100, 200, 10)]
for i in range(10):
    plt.scatter(b_feat[-1,i,:,0],
                    b_feat[-1,i,:,1],
                    s=20,
                    color=colors[i],
                    marker='o',
                    label='new classes',
                    edgecolor='none',
                    alpha=0.5)
    plt.scatter(b_proto[-1,i,0],
                    b_proto[-1,i,1],
                    s=20,
                    color='r',
                    marker='+',
                    label='prototype',
                    alpha=1)
# %%

def project(tensor, type='tsne'):
    if type == 'tsne':
        projector = TSNE(
            n_components=2,
            init='pca',
            learning_rate='auto',
            n_iter=3000,
            perplexity=25,
            verbose=True,
        )
    else:
        projector = PCA(n_components=2)
    return projector.fit_transform(tensor)


u_feat = upcl['features'].reshape(10,10,100,-1)
u_proto = upcl['prototype'].reshape(10,10,-1)

u_all_feat = np.concatenate((u_feat[0].reshape(-1,512),u_feat[-1].reshape(-1,512),u_proto[0],u_proto[-1]),0)

u_d_feat = project(u_all_feat.astype(np.double), 'tsne')

u_feat = u_d_feat[:2000].reshape(2,10,100,-1)
u_proto = u_d_feat[2000:].reshape(2,10,-1)

plt.figure()
colors = ['#65{:02x}08'.format(i) for i in range(100, 200, 10)]
for i in range(10):
    plt.scatter(u_feat[0,i,:,0],
                    u_feat[0,i,:,1],
                    s=20,
                    color=colors[i],
                    marker='o',
                    label='old classes',
                    edgecolor='none',
                    alpha=0.5)
    plt.scatter(u_proto[0,i,0],
                    u_proto[0,i,1],
                    s=20,
                    color='r',
                    marker='x',
                    label='prototype',
                    alpha=1)

colors = ['#0165{:02x}'.format(i) for i in range(100, 200, 10)]
for i in range(10):
    plt.scatter(u_feat[-1,i,:,0],
                    u_feat[-1,i,:,1],
                    s=20,
                    color=colors[i],
                    marker='o',
                    label='new classes',
                    edgecolor='none',
                    alpha=0.5)
    plt.scatter(u_proto[-1,i,0],
                    u_proto[-1,i,1],
                    s=20,
                    color='r',
                    marker='+',
                    label='prototype',
                    alpha=1)

# %%
plt.figure(figsize=(6, 6))
plt.subplot(212)
colors = similar_colors('#00CC00', 10)
for i in range(10):
    plt.scatter(b_feat[0,i,:,0],
                    b_feat[0,i,:,1],
                    s=20,
                    color=colors[i],
                    marker='o',
                    label='old classes',
                    edgecolor='none',
                    alpha=0.5)
    plt.scatter(b_proto[0,i,0],
                    b_proto[0,i,1],
                    s=20,
                    color='r',
                    marker='x',
                    label='prototype',
                    alpha=1)

colors = similar_colors('#1240AB', 10)
for i in range(10):
    plt.scatter(b_feat[-1,i,:,0],
                    b_feat[-1,i,:,1],
                    s=20,
                    color=colors[i],
                    marker='o',
                    label='new classes',
                    edgecolor='none',
                    alpha=0.5)
    plt.scatter(b_proto[-1,i,0],
                    b_proto[-1,i,1],
                    s=20,
                    color='r',
                    marker='+',
                    label='prototype',
                    alpha=1)
plt.title('UPCL')
plt.xticks([])
plt.yticks([])



plt.subplot(211)
colors = similar_colors('#00CC00', 10)
for i in range(10):
    plt.scatter(u_feat[0,i,:,0],
                    u_feat[0,i,:,1],
                    s=20,
                    color=colors[i],
                    marker='o',
                    label='old classes',
                    edgecolor='none',
                    alpha=0.5)
    plt.scatter(u_proto[0,i,0],
                    u_proto[0,i,1],
                    s=20,
                    color='r',
                    marker='x',
                    label='prototype',
                    alpha=1)

colors = similar_colors('#1240AB', 10)
for i in range(10):
    plt.scatter(u_feat[-1,i,:,0],
                    u_feat[-1,i,:,1],
                    s=20,
                    color=colors[i],
                    marker='o',
                    label='new classes',
                    edgecolor='none',
                    alpha=0.5)
    plt.scatter(u_proto[-1,i,0],
                    u_proto[-1,i,1],
                    s=20,
                    color='r',
                    marker='+',
                    label='prototype',
                    alpha=1)
plt.title('Baesline')
plt.xticks([])
plt.yticks([])
handles = [
    Line2D([0],[0],marker='+',color='r',linewidth=0),
    Line2D([0],[0],marker='x',color='r',linewidth=0),
    Line2D([0],[0],marker='o',color='#1240AB',linewidth=0),
    Line2D([0],[0],marker='o',color='#00CC00',linewidth=0),
    ]
plt.legend(handles=handles, labels=['old prototype', 'new prototype','old class','new class'],loc=5, bbox_to_anchor=(1,-0.1),borderaxespad = 0.)
plt.savefig('../logs/tsne.pdf', bbox_inches='tight')
# %%
