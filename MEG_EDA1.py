# %% [markdown]
# - 2024.06.18 データセットの詳細についての追記
#   - データセットの論文はこちらになります（データセットのリンクからもたどれます）．
#     - https://elifesciences.org/articles/82580
#   - 配布したデータは論文のセクション"MEG data preprocessing and cleaning"の内容が施されたものになります．
#     - よってサンプリングレートは200Hzです
#   - セクション"MEG data acquisition"に記載のある通り，チャンネルの座標系は[CTF 275 MEG system](https://mne.tools/1.6/auto_examples/visualization/meg_sensors.html#ctf)のものになっております．
#     - チャンネル座標をモデルに組み込みたい際の参考にしてください
#     - 配布データのチャンネル数が271しかない理由については，上記2セクションに記載のある通りです

# %% [markdown]
# raw = read_raw_ctf(
#     spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds"
# )
# fig = plot_alignment(raw.info, meg=("helmet", "sensors", "ref"), **kwargs)
# set_3d_title(figure=fig, title="CTF 275")

# %%
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample, spm_face, testing
from mne.preprocessing import (
    compute_proj_ecg,
    compute_proj_eog,
    create_ecg_epochs,
    create_eog_epochs,
)
from mne.io import (
    read_raw_artemis123,
    read_raw_bti,
    read_raw_ctf,
    read_raw_fif,
    read_raw_kit,
)
from mne.viz import plot_alignment, set_3d_title

from torch import t

data_dir='./data'
X = torch.load(os.path.join(data_dir, f"val_X.pt"))
y = torch.load(os.path.join(data_dir, f"val_y.pt"))
idx = torch.load(os.path.join(data_dir, f"val_subject_idxs.pt"))
#X = torch.load(os.path.join(data_dir, f"train_X.pt"))
#y = torch.load(os.path.join(data_dir, f"train_y.pt"))
#idx = torch.load(os.path.join(data_dir, f"train_subject_idxs.pt"))

# %%
X.shape

# %% [markdown]
# 271ch * 281data

# %%
y.shape

# %%
idx.shape

# %%
X[0]

# %%
y[0]

# %%
idx[0]

# %%
for i in range(5):
    plt.imshow(X[i], cmap='RdBu_r')
    plt.show()

# %%
for j in range(5):
    plt.plot(X[j].flatten(), alpha=0.7)

# %%
for j in range(5):
    plt.hist(X[j].flatten(), histtype='step', bins=50)
plt.xlim(-10, 10)

# %%
idx.unique()

# %% [markdown]
# idxは被験者と紐づいている=4名のデータ

# %%
df_idx=pd.DataFrame(idx)
df_idx.columns=['idx']
df_idx.groupby('idx')['idx'].count()

# %% [markdown]
# 1人当たり4108データ

# %%
y.unique()

# %% [markdown]
# 1854個の分類

# %%
df_y=pd.DataFrame(y)
df_y.columns=['y']
df_merge=pd.concat([df_idx, df_y], axis=1)
df_merge.groupby(['idx', 'y'])['y'].count().unique()

# %% [markdown]
# trainデータ: 1人当たり1カテゴリ8個、16個の2通り  
# validデータ: 1人当たり1カテゴリ2個、4個の2通り  

# %%
df_merge.groupby(['idx', 'y'])['y'].count().plot(kind='hist')

# %%
for i in np.arange(4108, 4108+5, 1):
    plt.imshow(X[i], cmap='RdBu_r')
    plt.show()

# %%
for j in np.arange(4108, 4108+5, 1):
    plt.plot(X[j].flatten(), alpha=0.7)

# %%
for j in np.arange(4108, 4108+5, 1):
    plt.hist(X[j].flatten(), histtype='step', bins=50)
plt.xlim(-10, 10)

# %% [markdown]
# ## MNE

# %% [markdown]
# [MNE](https://mne.tools/stable/index.html)  
# [MNEチュートリアル](https://qiita.com/sentencebird/items/035ba0c48569f06e3a42)  

# %% [markdown]
# Task Name: main  
# Manufacturer: CTF  
# Power Line Frequency: 60  
# Sampling Frequency: 1200  
# Software Filters: n/a  
# Recording Duration: 347.99916666666667  
# Recording Type: continuous  
# Dewar Position: n/a  
# Property is empty  
# Property is empty  
# M E G Channel Count: 272  
# M E G R E F Channel Count: 28  
# Property is empty  
# Property is empty  
# Property is empty  
# Property is empty  
# Misc Channel Count: 10  
# Property is empty  

# %%
n_channels=271
#sampling_freq= 281/1.9  # 281 samples in 1.9 second
sampling_freq= 200
info = mne.create_info(n_channels, sfreq=sampling_freq, ch_types="mag")
raw = mne.io.RawArray(X[0], info)
raw.load_data().resample(200)

# %%
kwargs = dict(eeg=False, coord_frame="meg", show_axes=True, verbose=True)
raw = read_raw_ctf(
    spm_face.data_path('./data') / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds"
)
fig = plot_alignment(raw.info, meg=("helmet", "sensors", "ref"), **kwargs)
set_3d_title(figure=fig, title="CTF 275")

# %%
raw[0]

# %%
raw.info

# %%
raw.plot(duration=0.348, n_channels=10, scalings={'mag': 1})
plt.show()

# %%
spectrum=raw.compute_psd()
for average in (False, True):
    spectrum.plot(
        average=average,
        dB=False,
        amplitude=True,
        xscale="log",
        picks="data",
        exclude="bads",
    )
plt.show()

# %%
raw_filt = raw.copy().filter(l_freq=1, h_freq=40)
raw_filt.plot(duration=0.348, n_channels=10, scalings={'mag': 1})
plt.show()
spectrum_filt=raw_filt.compute_psd()
for average in (False, True):
    spectrum_filt.plot(
        average=average,
        dB=False,
        amplitude=True,
        xscale="log",
        picks="data",
        exclude="bads",
    )
plt.show()

# %%
# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here

# %%
orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)

# %%
orig_raw.plot(duration=0.348)
raw.plot(duration=0.348)
plt.show()

# %%
from scipy.signal import stft
frequencies, times, Zxx = stft(X[5], fs=200, nperseg=281) # Short Time Fourier Transform
Zxx = np.abs(Zxx) # 振幅スペクトル (電極, 周波数, 時間)
times -= 0        # 時間領域を-1~2秒に合わせる

# %%
print(frequencies.shape, times.shape, Zxx.shape)

# %%
Zxx[0].shape

# %%
ERSP = 10*np.log(np.abs(Zxx))

# %%
cmap=plt.pcolormesh(times, frequencies, ERSP[0], shading='gouraud', cmap='Reds')
plt.ylim(0,100)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
#cmap.set_clim(0,0.3)
cbar=plt.colorbar(cmap)
cbar.ax.set_ylabel('Intensity [dB]')
plt.tight_layout()
plt.show()

# %%
rest = np.mean(ERSP[:,:,:len(times)//3], axis=2)[:,:,np.newaxis] #(電極, 周波数, 時間(タスク開始まで))の平均(電極, 周波数, 1)
ERD_ERS = (ERSP-rest)/rest*10

# %%
cmap=plt.pcolormesh(times, frequencies, ERD_ERS[0], shading='gouraud', cmap='RdBu_r')
plt.ylim(0,100)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
#cmap.set_clim(0,600)
cbar=plt.colorbar(cmap)
cbar.ax.set_ylabel('Intensity [dB]')
plt.tight_layout()
plt.show()

# %%
ERD_ERS_alpha = np.mean(ERD_ERS[:,(frequencies>=8)&(frequencies<=13)], axis=1) #(電極, 周波数(α帯域), 時間)の平均(電極, 時間)
ERD_ERS_beta = np.mean(ERD_ERS[:,(frequencies>=14)&(frequencies<=30)], axis=1) #(電極, 周波数(β帯域), 時間)の平均(電極, 時間)

# %%
info = mne.create_info(ch_names = raw.ch_names, sfreq = len(times)//(times[-1]-times[0]), ch_types='mag')

# %%
evoked = mne.EvokedArray(ERD_ERS_alpha, info, tmin=times[0])

# %%
evoked.set_montage(mne.channels.make_standard_montage('standard_1020'))

# %%
evoked.plot()
plt.show()

# %% [markdown]
# ## 主成分分析・独立成分分析

# %% [markdown]
# [参考HP1](https://qiita.com/yuta-takahashi/items/c05908db9aebd1afa99f)  
# [ICA参考HP2](https://qiita.com/maskot1977/items/ca40cb3983fdef3cc9df)  
# [PCA参考HP3](https://qiita.com/maskot1977/items/082557fcda78c4cdb41f)  

# %%
X[0,0].shape

# %% [markdown]
# Principal component analysis(主成分分析)

# %%
X[0].T.shape

# %%
from sklearn.decomposition import PCA

# PCAの実行
#pca=PCA(n_components=20, random_state=123)
pca=PCA(random_state=123)
feature=pca.fit_transform(X[0].T)
PCA_comp=pca.components_ #基底行列

#0番目のデータを20個のコンポーネントを用いて復元
plt.figure(figsize = (12, 2))
plt.plot(X[0,0], label="data") #0番目のデータ
plt.plot(np.dot(feature, PCA_comp).T[0], label="reconstruct") #復元した0番目のデータ
plt.legend()

# %%
X[0,:,0].shape

# %%
feature.shape

# %%
PCA_comp.shape

# %%
# 主成分得点
pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(X[0,:,0]))]).head()

# %%
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
plt.scatter(feature[:,0], feature[:,1], alpha=0.8, c=list(X[0,0,:]))
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %%
# 寄与率
pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(X[0,:,0]))])

# %%
# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.xlim(1,)
plt.xscale('log')
plt.grid()
plt.show()

# %%
# PCA の固有値
pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(X[0,:,0]))])

# %%
X[0,0,:].shape

# %%
# PCA の固有ベクトル
pd.DataFrame(pca.components_, columns=list(X[0,:,0]), index=["PC{}".format(x + 1) for x in range(len(X[0,:,0]))])

# %%
# 第一主成分と第二主成分における観測変数の寄与度をプロットする
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], list(X[0,0,:])):
    plt.text(x, y, name, alpha=0.3)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %% [markdown]
# Independent Component Analysis(独立成分分析)

# %%
X[0].T.shape

# %%
from sklearn.decomposition import FastICA

#ICAの実行
#ICA = FastICA(n_components=20, random_state=0)#20個の基底（コンポネント）を作る
ICA = FastICA(random_state=0)
X_transformed = ICA.fit_transform(X[0].T)
A_ =  ICA.mixing_ #混合行列

#0番目のデータを20個のコンポネントを用いて復元
plt.figure(figsize = (12, 2))
plt.plot(X[0,1], label="data")#0番目のデータ
plt.plot(np.dot(A_, X_transformed.T)[1], label="reconstruct")#復元した0番目のデータ
plt.legend()


# %%
np.dot(A_.T, X_transformed.T)[0].shape

# %%
X[0].shape

# %%
A_.shape

# %%
X_transformed.shape

# %%
t=np.linspace(0, 281/200 ,281)
x1 = np.array(X_transformed.T[0,:])
x2 = np.array(X_transformed.T[1,:])
x3 = np.array(X_transformed.T[2,:])

# %%
fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
ax1.plot(t, x1, color="green", label="Restored signal 1 (x1)")
ax2.plot(t, x2, color="green", label="Restored signal 2 (x2)")
ax3.plot(t, x3, color="green", label="Restored signal 3 (x3)")
ax1.legend(loc = 'upper right') 
ax2.legend(loc = 'upper right') 
ax3.legend(loc = 'upper right') 
fig.tight_layout()
plt.show()


